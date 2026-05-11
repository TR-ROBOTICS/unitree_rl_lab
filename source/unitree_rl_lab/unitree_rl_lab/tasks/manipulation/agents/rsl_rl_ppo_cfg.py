"""PPO runner config for valve-turn task.

Inherits all hyperparameters from locomotion BasePPORunnerCfg (PRD §PPO agent config).
Only experiment_name changed.
"""

from isaaclab.utils import configclass
from unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg import BasePPORunnerCfg


@configclass
class ValveTurnPPORunnerCfg(BasePPORunnerCfg):
    experiment_name = "valve_turn_g1_29dof"
