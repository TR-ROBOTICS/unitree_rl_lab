"""Manipulation MDP — task-specific reward, obs, event, termination functions."""

from .curriculums import (
    rim_distance_weight_anneal,
    turn_auto_curriculum_stage,
    turn_auto_curriculum_stage_easy,
    turn_smooth_curriculum_v5,
    turn_pd_curriculum_v6,
)
from .events import (
    reset_joints_to_fixed_pose,
    reset_valve_to_random_angle,
    reset_valve_base_position_random,
    reset_p_des_random,
    reset_arm_from_dataset,
    reset_arm_staged,
    reset_arm_v5_mixed,
    reset_arm_v6_mixed,
)
from .observations import (
    valve_pressure_now,
    valve_pressure_des,
    valve_pressure_des_random,
    valve_pos_robot_frame,
    last_arm_action,
)
from .rewards import (
    pressure_error,
    pressure_error_random,
    pressure_progress,
    pressure_progress_random,
    wheel_vel_toward_target,
    arm_joint_motion,
    rim_distance_reward,
    reach_approach_reward,
    reach_progress_reward,
    reach_hand_distance,
    reach_handoff_bonus,
    bilateral_contact,
    contact_force_jerk,
)
from .terminations import (
    joint_vel_runaway,
    pressure_success_hold,
    pressure_success_hold_random,
    reach_handoff_condition,
)
