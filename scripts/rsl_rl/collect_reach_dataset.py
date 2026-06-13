"""Collect reach-policy terminal arm states for Turn v3 bootstrap.

Runs the reach policy in play mode and captures arm joint positions at each
episode termination (timeout or handoff success).  Saves a (N, 14) float32
numpy array to ``datasets/reach_arm_positions.npy`` relative to the
unitree_rl_lab package root.

Usage:
    python scripts/rsl_rl/collect_reach_dataset.py \
        --task Unitree-G1-29dof-ValveReach-v0 \
        --num_envs 64 \
        --num_samples 2000 \
        --headless \
        --load_run saved_models \
        --checkpoint reach_model_200.pt

Dataset shape:  (num_samples, 14) — 14 arm joints in the order resolved by
    SceneEntityCfg("robot", joint_names=[".*_shoulder_.*", ".*_elbow_.*", ".*_wrist_.*"]).
    Same order expected by reset_arm_from_dataset EventTerm in Turn v3.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import pathlib
import re

from importlib.metadata import version
from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Collect reach terminal arm states.")
parser.add_argument("--num_envs", type=int, default=64, help="Parallel envs.")
parser.add_argument("--num_samples", type=int, default=2000, help="Target dataset size.")
parser.add_argument("--task", type=str, default="Unitree-G1-29dof-ValveReach-v0")
parser.add_argument("--disable_fabric", action="store_true", default=False)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os

import gymnasium as gym
import numpy as np
import torch
from tqdm import tqdm

from rsl_rl.runners import OnPolicyRunner

import isaaclab_tasks  # noqa: F401
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from isaaclab_tasks.utils import get_checkpoint_path

import unitree_rl_lab.tasks  # noqa: F401
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg

# Arm joint name patterns — must match action space definition.
_ARM_PATTERNS = [".*_shoulder_.*", ".*_elbow_.*", ".*_wrist_.*"]


def resolve_arm_joint_ids(robot_joint_names: list[str]) -> list[int]:
    """Return indices of arm joints in robot.joint_names, in their USD order."""
    ids = []
    for i, name in enumerate(robot_joint_names):
        for pat in _ARM_PATTERNS:
            if re.fullmatch(pat, name):
                ids.append(i)
                break
    return ids


def main():
    # ---- env + policy setup (mirrors play.py) ----
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
        entry_point_key="play_env_cfg_entry_point",
    )
    # Override num_envs from CLI
    env_cfg.scene.num_envs = args_cli.num_envs

    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, "5.0.1")

    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))

    # When --load_run is provided, resolve via get_checkpoint_path (handles the
    # logs/rsl_rl/<exp>/<load_run>/<checkpoint> layout, including symlinks).
    # retrieve_file_path only works for absolute paths / nucleus URLs.
    if agent_cfg.load_run:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    print(f"[collect_reach_dataset] Loading checkpoint: {resume_path}")

    env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # ---- resolve arm joint IDs once ----
    robot = env.unwrapped.scene["robot"]
    arm_ids = resolve_arm_joint_ids(robot.joint_names)
    print(f"[collect_reach_dataset] Arm joint count: {len(arm_ids)}")
    assert len(arm_ids) == 14, f"Expected 14 arm joints, got {len(arm_ids)}"

    arm_ids_t = torch.tensor(arm_ids, device=env.unwrapped.device, dtype=torch.long)

    # ---- collection loop ----
    terminal_states: list[np.ndarray] = []
    target = args_cli.num_samples

    obs = env.get_observations()
    if version("rsl-rl-lib").startswith("2.3."):
        obs, _ = env.get_observations()

    print(f"[collect_reach_dataset] Collecting {target} terminal states...", flush=True)

    with tqdm(total=target, unit="samples", dynamic_ncols=True) as pbar:
        while simulation_app.is_running() and len(terminal_states) < target:
            # Capture arm joint state BEFORE step — terminal state for any env
            # that terminates on this step.
            pre_step_joints = robot.data.joint_pos[:, arm_ids_t].clone()  # (num_envs, 14)

            with torch.inference_mode():
                actions = policy(obs)
                obs, _, dones, _ = env.step(actions)

            # dones: (num_envs,) bool — True for envs that just terminated + reset
            done_ids = dones.nonzero(as_tuple=True)[0]
            new = len(done_ids)
            if new:
                for i in done_ids:
                    terminal_states.append(pre_step_joints[i].cpu().numpy())
                pbar.update(min(new, target - pbar.n))

    # ---- save ----
    dataset = np.stack(terminal_states[:target]).astype(np.float32)
    print(f"[collect_reach_dataset] Dataset shape: {dataset.shape}")

    # Output path: unitree_rl_lab package root / datasets /
    pkg_root = pathlib.Path(__file__).parents[2] / "source" / "unitree_rl_lab" / "unitree_rl_lab"
    out_dir = pkg_root / "datasets"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "reach_arm_positions.npy"
    np.save(str(out_path), dataset)
    print(f"[collect_reach_dataset] Saved to: {out_path}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
