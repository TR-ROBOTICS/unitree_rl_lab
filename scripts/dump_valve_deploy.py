"""Dump valve-turn deploy ground truth: articulation joint order, arm action
term joint_ids/order, full default (pregrip) pose, valve joint name.

Resolves open item #1 of the deploy contract — the Isaac arm-DOF order that the
ONNX action vector follows, mapped to G1 SDK motor indices. Run headless:

    cd ~/unitree_rl_lab && conda run -n env_isaaclab python scripts/dump_valve_deploy.py \
        --task Unitree-G1-29dof-ValveTurn-v5 --headless --num_envs 1
"""

import argparse
import json

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=1)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym

import isaaclab_tasks  # noqa: F401
import unitree_rl_lab.tasks  # noqa: F401
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg

# G1 SDK motor enum (defines.h) — left arm 15-21, right arm 22-28.
SDK_INDEX = {
    "left_shoulder_pitch_joint": 15, "left_shoulder_roll_joint": 16,
    "left_shoulder_yaw_joint": 17, "left_elbow_joint": 18,
    "left_wrist_roll_joint": 19, "left_wrist_pitch_joint": 20,
    "left_wrist_yaw_joint": 21,
    "right_shoulder_pitch_joint": 22, "right_shoulder_roll_joint": 23,
    "right_shoulder_yaw_joint": 24, "right_elbow_joint": 25,
    "right_wrist_roll_joint": 26, "right_wrist_pitch_joint": 27,
    "right_wrist_yaw_joint": 28,
    # body
    "left_hip_pitch_joint": 0, "left_hip_roll_joint": 1, "left_hip_yaw_joint": 2,
    "left_knee_joint": 3, "left_ankle_pitch_joint": 4, "left_ankle_roll_joint": 5,
    "right_hip_pitch_joint": 6, "right_hip_roll_joint": 7, "right_hip_yaw_joint": 8,
    "right_knee_joint": 9, "right_ankle_pitch_joint": 10, "right_ankle_roll_joint": 11,
    "waist_yaw_joint": 12, "waist_roll_joint": 13, "waist_pitch_joint": 14,
}


def main():
    env_cfg = parse_env_cfg(args_cli.task, num_envs=args_cli.num_envs)
    env = gym.make(args_cli.task, cfg=env_cfg)
    env.reset()
    u = env.unwrapped

    robot = u.scene["robot"]
    jnames = list(robot.data.joint_names)
    dpos = robot.data.default_joint_pos[0].cpu().numpy().tolist()

    arm_term = u.action_manager._terms["arm"]
    arm_ids = list(arm_term._joint_ids) if not isinstance(arm_term._joint_ids, slice) else list(range(len(jnames)))
    # _joint_names is the resolved order the action vector follows
    arm_names = list(getattr(arm_term, "_joint_names", [jnames[i] for i in arm_ids]))

    valve = u.scene["valve_rig"]
    out = {
        "task": args_cli.task,
        "robot_joint_names_in_order": jnames,
        "robot_default_joint_pos": dpos,
        "arm_action_joint_ids": arm_ids,
        "arm_action_joint_names_in_order": arm_names,
        "valve_joint_names": list(valve.data.joint_names),
        # the deploy mapping we actually need:
        "arm_isaac_to_sdk": [
            {"isaac_idx": k, "joint": nm, "sdk_idx": SDK_INDEX.get(nm),
             "default": dpos[arm_ids[k]] if k < len(arm_ids) else None}
            for k, nm in enumerate(arm_names)
        ],
        "full_isaac_to_sdk": [
            {"isaac_idx": i, "joint": nm, "sdk_idx": SDK_INDEX.get(nm), "default": dpos[i]}
            for i, nm in enumerate(jnames)
        ],
    }
    print("===DUMP_BEGIN===")
    print(json.dumps(out, indent=2))
    print("===DUMP_END===")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
