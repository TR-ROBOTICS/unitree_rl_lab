import gymnasium as gym

# Stage 1 — turn policy (fixed θ_init=θ_min, fixed p_des=50 PSI).
# Entry point resolves through the shim dof29/valve_turn_env_cfg.py →
# valve/turn_env_cfg.py.  ID preserved.
gym.register(
    id="Unitree-G1-29dof-ValveTurn-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.valve_turn_env_cfg:ValveTurnEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.valve_turn_env_cfg:ValveTurnPlayEnvCfg",
        "rsl_rl_cfg_entry_point": (
            "unitree_rl_lab.tasks.manipulation.agents.rsl_rl_ppo_cfg:ValveTurnPPORunnerCfg"
        ),
    },
)

# Stage 2 — turn policy v1 (rim-distance reward, smoothness curriculum scaffold).
# Entry point resolves through the shim dof29/valve_turn_env_cfg_v1.py →
# valve/turn_env_cfg_v1.py.  ID preserved.
gym.register(
    id="Unitree-G1-29dof-ValveTurn-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.valve_turn_env_cfg_v1:ValveTurnEnvCfgV1",
        "play_env_cfg_entry_point": f"{__name__}.valve_turn_env_cfg_v1:ValveTurnPlayEnvCfgV1",
        "rsl_rl_cfg_entry_point": (
            "unitree_rl_lab.tasks.manipulation.agents.rsl_rl_ppo_cfg:ValveTurnPPORunnerCfg"
        ),
    },
)

# Stage 2 v2 — random θ_init + random p_des per episode.
gym.register(
    id="Unitree-G1-29dof-ValveTurn-v2",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.valve_turn_env_cfg_v2:ValveTurnEnvCfgV2",
        "play_env_cfg_entry_point": f"{__name__}.valve_turn_env_cfg_v2:ValveTurnPlayEnvCfgV2",
        "rsl_rl_cfg_entry_point": (
            "unitree_rl_lab.tasks.manipulation.agents.rsl_rl_ppo_cfg:ValveTurnPPORunnerCfg"
        ),
    },
)

# Stage 2 v4 — v3 + smoothness penalty (action_rate_l2 = -0.005).
gym.register(
    id="Unitree-G1-29dof-ValveTurn-v4",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.valve_turn_env_cfg_v4:ValveTurnEnvCfgV4",
        "play_env_cfg_entry_point": f"{__name__}.valve_turn_env_cfg_v4:ValveTurnPlayEnvCfgV4",
        "rsl_rl_cfg_entry_point": (
            "unitree_rl_lab.tasks.manipulation.agents.rsl_rl_ppo_cfg:ValveTurnPPORunnerCfg"
        ),
    },
)

# Stage 2 v3 — arm init DR from reach terminal-state dataset.
# Falls back to pre-grip pose if dataset absent (training can start before collection).
gym.register(
    id="Unitree-G1-29dof-ValveTurn-v3",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.valve_turn_env_cfg_v3:ValveTurnEnvCfgV3",
        "play_env_cfg_entry_point": f"{__name__}.valve_turn_env_cfg_v3:ValveTurnPlayEnvCfgV3",
        "rsl_rl_cfg_entry_point": (
            "unitree_rl_lab.tasks.manipulation.agents.rsl_rl_ppo_cfg:ValveTurnPPORunnerCfg"
        ),
    },
)

# Turn v4a — 3-stage auto-curriculum, dataset arm init from Stage 0 (hard).
# Stages: 0=fixed θ/p+dataset, 1=random θ, 2=random p.  window_iters=100.
gym.register(
    id="Unitree-G1-29dof-ValveTurn-v4a",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.valve_turn_env_cfg_v4a:ValveTurnEnvCfgV4A",
        "play_env_cfg_entry_point": f"{__name__}.valve_turn_env_cfg_v4a:ValveTurnPlayEnvCfgV4A",
        "rsl_rl_cfg_entry_point": (
            "unitree_rl_lab.tasks.manipulation.agents.rsl_rl_ppo_cfg:ValveTurnPPORunnerCfg"
        ),
    },
)

# Turn v4ae — 4-stage auto-curriculum, pre-grip Stage 0 (easy). Mirrors v0→v3 chain.
# Stages: 0=fixed θ/p+pregrasp, 1=random θ, 2=random p, 3=dataset arm.  window_iters=100.
gym.register(
    id="Unitree-G1-29dof-ValveTurn-v4ae",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.valve_turn_env_cfg_v4ae:ValveTurnEnvCfgV4AE",
        "play_env_cfg_entry_point": f"{__name__}.valve_turn_env_cfg_v4ae:ValveTurnPlayEnvCfgV4AE",
        "rsl_rl_cfg_entry_point": (
            "unitree_rl_lab.tasks.manipulation.agents.rsl_rl_ppo_cfg:ValveTurnPPORunnerCfg"
        ),
    },
)

# Turn v4ah — v4a + 38-DoF action (arm+hands), obs 92d. Train from scratch.
# Same 3-stage auto-curriculum as v4a (dataset arm Stage 0). Fingers active from iter 0.
gym.register(
    id="Unitree-G1-29dof-ValveTurn-v4ah",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.valve_turn_env_cfg_v4ah:ValveTurnEnvCfgV4AH",
        "play_env_cfg_entry_point": f"{__name__}.valve_turn_env_cfg_v4ah:ValveTurnPlayEnvCfgV4AH",
        "rsl_rl_cfg_entry_point": (
            "unitree_rl_lab.tasks.manipulation.agents.rsl_rl_ppo_cfg:ValveTurnPPORunnerCfg"
        ),
    },
)

# Turn v4aeh — v4ae + 38-DoF action (arm+hands), obs 92d. Train from scratch.
# Same 4-stage auto-curriculum as v4ae. Fingers active from iter 0.
gym.register(
    id="Unitree-G1-29dof-ValveTurn-v4aeh",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.valve_turn_env_cfg_v4aeh:ValveTurnEnvCfgV4AEH",
        "play_env_cfg_entry_point": f"{__name__}.valve_turn_env_cfg_v4aeh:ValveTurnPlayEnvCfgV4AEH",
        "rsl_rl_cfg_entry_point": (
            "unitree_rl_lab.tasks.manipulation.agents.rsl_rl_ppo_cfg:ValveTurnPPORunnerCfg"
        ),
    },
)

# Turn v5 — smooth dual-axis curriculum + bimanual contact + CCD fingers. Fresh init.
# Stage 0: θ+p_des expand simultaneously. Stage 1: dataset arm mixing. Stage 2: full.
gym.register(
    id="Unitree-G1-29dof-ValveTurn-v5",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.valve.turn_env_cfg_v5:ValveTurnEnvCfgV5",
        "play_env_cfg_entry_point": f"{__name__}.valve.turn_env_cfg_v5:ValveTurnPlayEnvCfgV5",
        "rsl_rl_cfg_entry_point": (
            "unitree_rl_lab.tasks.manipulation.agents.rsl_rl_ppo_cfg:ValveTurnPPORunnerCfg"
        ),
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
