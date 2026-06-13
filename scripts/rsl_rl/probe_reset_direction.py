"""Reset direction-balance probe — verifies the centered-p_des fix.

Read-only diagnostic: no training, no checkpoint writes, no file saves.

Background
---------
Turn direction at reset = sign(θ_des − θ_init):
  θ_init ~ Uniform[θ_min, θ_max]   (reset_valve_to_random_angle, events.py)
  θ_des  = (p_des − b) / a         (inverse of g(θ) = a·θ + b)
  θ_init > θ_des → CW  (decrease θ to reach target)
  θ_init < θ_des → CCW (increase θ)

With the old fixed p_des = 50 PSI the target sat near θ_min, so ~81 % of resets
were CW / 19 % CCW — a bias that degraded CCW turn quality downstream
(see docs/model_eval_qual.md). Centering p_des → 107 PSI puts θ_des at the
midpoint of [θ_min, θ_max], which should make the split ~50/50 with equal mean
|Δθ|. This probe measures the *actual* env reset to confirm that.

Usage (from /home/jescobars/unitree_rl_lab/scripts/rsl_rl/):
    conda activate env_isaaclab
    python probe_reset_direction.py --task Unitree-G1-29dof-ValveTurn-v1 --headless
    python probe_reset_direction.py --task Unitree-G1-29dof-ValveTurn-v2 --headless

v1 = fixed p_des (the stage the fix targets) → expect ~50/50 after the fix.
v2 = random p_des ∈ [15,200] → marginal split is already ~50/50 by construction.

Exit code 0 if the CCW fraction is within --tol of 0.50, else 1.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import importlib

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Valve reset direction-balance probe.")
parser.add_argument("--task", type=str, default="Unitree-G1-29dof-ValveTurn-v1",
                    help="gym task id (use a fixed-p_des stage like ...-v1 to test the fix).")
parser.add_argument("--num_envs", type=int, default=4096,
                    help="envs reset in parallel = sample size for one reset.")
parser.add_argument("--tol", type=float, default=0.05,
                    help="pass if |CCW_fraction - 0.5| <= tol.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

import gymnasium as gym

importlib.import_module("isaaclab_tasks")          # registers isaaclab tasks
importlib.import_module("unitree_rl_lab.tasks")    # registers Unitree-* gym ids

from unitree_rl_lab.utils.parser_cfg import parse_env_cfg


def main() -> int:
    print(f"\n{'='*64}")
    print("Valve reset direction-balance probe")
    print(f"Task:     {args_cli.task}")
    print(f"num_envs: {args_cli.num_envs}  (one reset = {args_cli.num_envs} samples)")
    print(f"{'='*64}\n")

    env_cfg = parse_env_cfg(
        args_cli.task,
        device="cuda:0",
        num_envs=args_cli.num_envs,
        use_fabric=True,
        entry_point_key="env_cfg_entry_point",
    )
    env_cfg.scene.num_envs = args_cli.num_envs

    # g(θ) coefficients + θ bounds are explicit configclass fields (firmware-locked).
    a = float(env_cfg.pressure_a)
    b = float(env_cfg.pressure_b)
    theta_min = float(env_cfg.theta_min)
    theta_max = float(env_cfg.theta_max)
    p_des_lo, p_des_hi = (float(x) for x in env_cfg.p_des_range)

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = env.unwrapped

    obs_dict, _ = env.reset()

    # θ_init: valve RevoluteJoint (index 0) position per env, post-reset.
    valve_rig = env.scene["valve_rig"]
    theta_init = valve_rig.data.joint_pos[:, 0].clone()  # (num_envs,)

    # p_des per env: v2+ stores a per-env buffer; v0/v1 use the fixed cfg value.
    if hasattr(env, "p_des_buf"):
        p_des = env.p_des_buf.reshape(-1)[: theta_init.numel()].to(theta_init.device).float()
        p_des_src = "env.p_des_buf (per-env)"
    else:
        p_des = torch.full_like(theta_init, p_des_lo)
        p_des_src = f"cfg p_des_range[0] = {p_des_lo:.1f} PSI (fixed)"

    # θ_des = inverse g(θ), clamped to achievable θ range.
    theta_des = ((p_des - b) / a).clamp(theta_min, theta_max)

    delta = theta_des - theta_init          # >0 → CCW (increase θ), <0 → CW
    ccw = delta > 0
    cw = delta < 0
    n = theta_init.numel()
    n_ccw = int(ccw.sum().item())
    n_cw = int(cw.sum().item())
    frac_ccw = n_ccw / n

    mean_mag_ccw = float(delta[ccw].abs().mean().item()) if n_ccw else 0.0
    mean_mag_cw = float(delta[cw].abs().mean().item()) if n_cw else 0.0

    print("--- reset distribution ---")
    print(f"g(θ): p = {a:.3f}·θ + {b:.2f} PSI   θ∈[{theta_min:.2f}, {theta_max:.2f}] rad")
    print(f"p_des source: {p_des_src}")
    if hasattr(env, "p_des_buf"):
        print(f"p_des sampled:  [{float(p_des.min()):.1f}, {float(p_des.max()):.1f}] PSI "
              f"(cfg range [{p_des_lo:.1f}, {p_des_hi:.1f}])")
        print(f"θ_des sampled:  [{float(theta_des.min()):.2f}, {float(theta_des.max()):.2f}] rad")
    else:
        print(f"θ_des (fixed):  {float(theta_des[0]):.2f} rad")
    print(f"θ_init sampled: [{float(theta_init.min()):.2f}, {float(theta_init.max()):.2f}] rad  "
          f"mean {float(theta_init.mean()):.2f}")
    print()
    print(f"CCW (turn θ up):   {n_ccw:5d} / {n}  = {frac_ccw*100:5.1f} %   mean |Δθ| = {mean_mag_ccw:5.2f} rad")
    print(f"CW  (turn θ down): {n_cw:5d} / {n}  = {(n_cw/n)*100:5.1f} %   mean |Δθ| = {mean_mag_cw:5.2f} rad")
    print()

    off = abs(frac_ccw - 0.5)
    ok = off <= args_cli.tol
    print(f"balance offset |CCW - 0.50| = {off:.3f}   tol = {args_cli.tol:.3f}   "
          f"=> {'PASS' if ok else 'FAIL'}")
    print(f"{'='*64}\n")

    env.close()
    return 0 if ok else 1


if __name__ == "__main__":
    rc = main()
    simulation_app.close()
    raise SystemExit(rc)
