import gymnasium as gym

# Turn policies — every experiment is a row in valve/presets.py TURN_PRESETS.
# Gym ID "Unitree-G1-29dof-ValveTurn-<name>" resolves to the per-preset factory
# callables presets:turn_<name> (train) / presets:turn_<name>_play (single-env).
# IDs unchanged from the former subclass-ladder registrations (ADR 0009 item 5).
# Names hardcoded (mirror TURN_PRESETS keys) to keep registration lazy — the heavy
# cfg module is imported only when gym.make resolves an entry_point, after the
# Sim app is launched, not at task-package import time.
_TURN_PRESET_NAMES = ("v0", "v1", "v2", "v3", "v4", "v4a", "v4ae", "v4ah", "v4aeh", "v5", "v6", "v6b", "v7", "v7_palmcaps", "v7_nocaps")
_RSL_RL_TURN = "unitree_rl_lab.tasks.manipulation.agents.rsl_rl_ppo_cfg:ValveTurnPPORunnerCfg"

for _name in _TURN_PRESET_NAMES:
    gym.register(
        id=f"Unitree-G1-29dof-ValveTurn-{_name}",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.valve.presets:turn_{_name}",
            "play_env_cfg_entry_point": f"{__name__}.valve.presets:turn_{_name}_play",
            "rsl_rl_cfg_entry_point": _RSL_RL_TURN,
        },
    )

# Reach policy — two-policy chain Phase 1 (ADR 0004).
# 45d obs: joint_pos(14) + joint_vel(14) + valve_pos(3) + last_action(14).
# Handoff termination: |Δx|<1cm AND √(Δy²+Δz²)<3cm both hands.
gym.register(
    id="Unitree-G1-29dof-ValveReach-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.valve.reach_env_cfg:ValveReachEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.valve.reach_env_cfg:ValveReachPlayEnvCfg",
        "rsl_rl_cfg_entry_point": (
            "unitree_rl_lab.tasks.manipulation.agents.rsl_rl_ppo_cfg:ValveReachPPORunnerCfg"
        ),
    },
)
