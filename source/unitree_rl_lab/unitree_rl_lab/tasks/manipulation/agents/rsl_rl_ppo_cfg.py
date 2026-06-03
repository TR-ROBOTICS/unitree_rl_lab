"""PPO runner config for valve-turn task.

Manipulation-specific overrides on top of BasePPORunnerCfg:
- normalize_advantage=True  — stabilises training with bimodal contact rewards
- value_loss_coef reduced    — critic less dominant when reward variance high
- wheel_vel weight reduced    — see RewardsCfg
"""

from isaaclab.utils import configclass
from rsl_rl.runners import OnPolicyRunner
from unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg import BasePPORunnerCfg


@configclass
class ValveTurnPPORunnerCfg(BasePPORunnerCfg):
    experiment_name = "valve_turn_g1_29dof"

    def __post_init__(self):
        super().__post_init__()
        # Stabilise value loss with bimodal contact reward distribution
        self.algorithm.value_loss_coef = 0.5      # was 1.0
        self.algorithm.normalize_advantage_per_mini_batch = True  # stabilise bimodal rewards


@configclass
class ValveReachPPORunnerCfg(BasePPORunnerCfg):
    """PPO runner config for the reach policy (two-policy chain Phase 1, ADR 0004).

    Reach reward is purely distance-based (no contact instability) → same
    value_loss_coef and advantage normalisation as turn, but kept separate
    so both can be tuned independently.
    """

    experiment_name = "valve_reach_g1_29dof"

    def __post_init__(self):
        super().__post_init__()
        self.algorithm.value_loss_coef = 0.5
        self.algorithm.normalize_advantage_per_mini_batch = True
        self.algorithm.entropy_coef = 0.001  # default 0.01 → std diverges when reward gradient ≈ 0 at d>1m
