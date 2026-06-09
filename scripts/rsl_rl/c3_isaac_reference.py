"""C3 Isaac reference rollout — hand/valve world positions at reset + policy steps.

Read-only diagnostic: no training, no checkpoint writes, no file saves.

Task: Unitree-G1-29dof-ValveTurn-v4 (dataset arm init, random θ/p_des).
Policy: exported JIT from logs/rsl_rl/valve_turn_g1_29dof/2026-05-27_17-08-10/exported/policy.pt

Reports:
  1. World positions of left_hand_base_link, right_hand_base_link, valve hub at reset.
  2. Arm joint order (14d) — resolves left-block-then-right vs interleaved.
  3. Reset arm qpos (14d).
  4. Step 0 obs (30d) + action (14d).
  5. Hand world pos + valve theta every 10 steps for 50 steps.

Usage (from /home/jescobars/unitree_rl_lab/scripts/rsl_rl/):
    conda activate env_isaaclab
    python c3_isaac_reference.py --headless

The script forces the dataset init to use a fixed row (index 0) so results are
reproducible.  It also forces p_des=100 PSI (mid-range) for a clear turning
direction.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="C3 Isaac reference rollout.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch
import numpy as np
import gymnasium as gym

import isaaclab_tasks  # noqa: F401
import unitree_rl_lab.tasks  # noqa: F401  — registers Unitree-* gym ids

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
_TASK_ID   = "Unitree-G1-29dof-ValveTurn-v4"
_POLICY_PT = (
    "/home/jescobars/unitree_rl_lab/"
    "logs/rsl_rl/valve_turn_g1_29dof/2026-05-27_17-08-10/exported/policy.pt"
)
_DATASET_PT = (
    "/home/jescobars/unitree_rl_lab/source/unitree_rl_lab/"
    "unitree_rl_lab/datasets/reach_arm_positions.npy"
)
_P_DES_PSI   = 100.0    # target pressure — 100 PSI (mid-range, θ_des ≈ 28.1 rad)
_P_SPAN      = 185.0
_NUM_STEPS   = 50
_REPORT_EVERY = 10

# Hub body name confirmed via Script Editor 2026-05-20 (base_cfg.py:_HUB_BODY_NAME)
_HUB_BODY    = "mesh_50_AL_250_B7_8_A_stl"
_L_HAND_BODY = "left_hand_base_link"
_R_HAND_BODY = "right_hand_base_link"

# Arm joint select pattern
_ARM_PATTERNS = [".*_shoulder_.*", ".*_elbow_.*", ".*_wrist_.*"]


def match_arm(name: str) -> bool:
    import re
    return any(re.fullmatch(p, name) for p in _ARM_PATTERNS)


def body_world_pos(asset, body_name: str) -> torch.Tensor:
    """Return world-frame position (3,) for a named body on an articulation, env 0.

    Uses data.body_pos_w: (num_envs, num_bodies, 3).
    find_bodies returns (list[int], list[str]).
    """
    ids, _ = asset.find_bodies(body_name)
    return asset.data.body_pos_w[0, ids[0], :]


def get_valve_theta(env) -> float:
    """Read valve joint position (rad), env 0."""
    valve_rig = env.scene["valve_rig"]
    # joint 0 of valve_rig is RevoluteJoint
    return float(valve_rig.data.joint_pos[0, 0].item())


def main():
    print(f"\n{'='*60}")
    print(f"C3 Isaac reference rollout")
    print(f"Task:   {_TASK_ID}")
    print(f"Policy: {_POLICY_PT}")
    print(f"{'='*60}\n")

    # ---- Load dataset row 0 to see what arm init looks like ----
    dataset = np.load(_DATASET_PT)
    print(f"Dataset shape: {dataset.shape}  (expect 10000×14)")
    row0 = dataset[0]
    print(f"Dataset row 0 (arm qpos, rad): {np.round(row0, 4)}")

    # ---- Build env config ----
    from unitree_rl_lab.utils.parser_cfg import parse_env_cfg
    env_cfg = parse_env_cfg(
        _TASK_ID,
        device="cuda:0",
        num_envs=1,
        use_fabric=True,
        entry_point_key="play_env_cfg_entry_point",
    )
    env_cfg.scene.num_envs = 1

    # Override p_des to fixed 100 PSI for reproducibility — patch Stage 1 obs
    # (v4 inherits v2 random p_des from env buffer; we'll patch after reset).
    # Physics / reward config irrelevant for read-only diagnostic.

    # ---- Create env ----
    env = gym.make(_TASK_ID, cfg=env_cfg)
    env = env.unwrapped   # peel gym wrappers → ManagerBasedRLEnv

    # ---- Load JIT policy ----
    policy = torch.jit.load(_POLICY_PT).to("cuda:0")
    policy.eval()
    print(f"\nPolicy loaded: {_POLICY_PT}")

    # ---- Identify arm joint indices in robot ----
    robot = env.scene["robot"]
    all_joint_names = robot.joint_names
    arm_indices = [i for i, n in enumerate(all_joint_names) if match_arm(n)]
    arm_names   = [all_joint_names[i] for i in arm_indices]

    print(f"\n--- Robot joint count: {len(all_joint_names)} ---")
    print("Arm joint order (14d) in Isaac articulation:")
    for k, (idx, name) in enumerate(zip(arm_indices, arm_names)):
        print(f"  [{k:2d}] robot_joint[{idx:3d}]  {name}")

    assert len(arm_indices) == 14, f"Expected 14 arm joints, got {len(arm_indices)}"

    # ---- Reset ----
    obs_dict, _ = env.reset()
    obs = obs_dict["policy"]   # (1, 30)

    # Inject deterministic p_des into the env buffer (v2+ stores p_des_buf).
    # This ensures the pressure-error obs component is meaningful.
    if hasattr(env, "p_des_buf"):
        env.p_des_buf[:] = _P_DES_PSI
        print(f"p_des_buf patched → {_P_DES_PSI} PSI")
    else:
        print("p_des_buf not found — p_des from episode random draw")

    # ---- World positions at reset ----
    valve_rig = env.scene["valve_rig"]

    def report_positions(label: str):
        theta = get_valve_theta(env)
        p_now_psi = float(np.clip(4.527 * theta - 27.66, 15.0, 200.0))

        # Hand base links
        l_pos = body_world_pos(robot, _L_HAND_BODY)
        r_pos = body_world_pos(robot, _R_HAND_BODY)

        # Valve hub
        hub_idx = valve_rig.find_bodies(_HUB_BODY)[0]
        hub_pos = valve_rig.data.body_state_w[0, hub_idx[0], :3]

        dist_l = float(torch.norm(l_pos - hub_pos).item())
        dist_r = float(torch.norm(r_pos - hub_pos).item())

        print(f"\n[{label}]")
        print(f"  valve θ = {theta:.4f} rad  |  p_now ≈ {p_now_psi:.1f} PSI")
        print(f"  hub       world xyz: ({hub_pos[0]:.4f}, {hub_pos[1]:.4f}, {hub_pos[2]:.4f})")
        print(f"  L_hand    world xyz: ({l_pos[0]:.4f}, {l_pos[1]:.4f}, {l_pos[2]:.4f})  dist_to_hub={dist_l:.4f} m")
        print(f"  R_hand    world xyz: ({r_pos[0]:.4f}, {r_pos[1]:.4f}, {r_pos[2]:.4f})  dist_to_hub={dist_r:.4f} m")
        print(f"  L_hand x≈ {l_pos[0]:.3f}  R_hand x≈ {r_pos[0]:.3f}  hub x≈ {hub_pos[0]:.3f}")
        return theta, l_pos.cpu().numpy(), r_pos.cpu().numpy()

    # ---- Arm qpos at reset ----
    arm_qpos = robot.data.joint_pos[0, arm_indices].cpu().numpy()
    print(f"\n--- Arm qpos at reset (14d, rad) ---")
    for k, (name, q) in enumerate(zip(arm_names, arm_qpos)):
        print(f"  [{k:2d}] {name:<40s} = {q:+.4f}")

    # Compare to dataset row 0
    diff = arm_qpos - row0
    print(f"\nDiff vs dataset row 0: {np.round(diff, 4)}")
    print(f"Max abs diff: {np.max(np.abs(diff)):.4f} rad")

    report_positions("RESET")

    # ---- Step 0 obs + action ----
    obs_np = obs[0].cpu().numpy()
    print(f"\n--- Step 0 obs (30d) ---")
    print(f"  [0:14]  joint_pos_rel : {np.round(obs_np[0:14], 4)}")
    print(f"  [14:28] joint_vel_rel : {np.round(obs_np[14:28], 4)}")
    print(f"  [28]    p_now_norm    : {obs_np[28]:.4f}  (={obs_np[28]*185+15:.1f} PSI raw ≈ {obs_np[28]*185:.1f} span)")
    print(f"  [29]    p_des_norm    : {obs_np[29]:.4f}  (={obs_np[29]*185:.1f} PSI span)")

    with torch.inference_mode():
        action0 = policy(obs)
    action0_np = action0[0].cpu().numpy()
    print(f"\n--- Step 0 action (14d, pre-scale) ---")
    print(f"  raw net output: {np.round(action0_np, 4)}")
    print(f"  ×0.1 (rad Δ):   {np.round(action0_np * 0.1, 4)}")
    print(f"  action norm:    {np.linalg.norm(action0_np):.4f}")

    # ---- Step loop — 50 steps ----
    print(f"\n{'='*60}")
    print(f"Stepping {_NUM_STEPS} steps (reporting every {_REPORT_EVERY})...")
    print(f"{'='*60}")

    for step in range(1, _NUM_STEPS + 1):
        with torch.inference_mode():
            # Scale action to joint position targets (sim uses +Δ convention)
            obs_dict2, _, terminated, truncated, _ = env.step(action0)
            obs = obs_dict2["policy"]
            action0 = policy(obs)

        if step % _REPORT_EVERY == 0:
            report_positions(f"step {step:3d}")
            act_np = action0[0].cpu().numpy()
            print(f"  action norm: {np.linalg.norm(act_np):.4f}  "
                  f"max|a|: {np.max(np.abs(act_np)):.4f}")
            if terminated[0] or truncated[0]:
                print(f"  *** Episode terminated at step {step} ***")
                break

    print(f"\n{'='*60}")
    print("C3 reference rollout complete.")
    print(f"{'='*60}\n")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
