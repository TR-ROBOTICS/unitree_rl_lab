import gymnasium as gym

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
