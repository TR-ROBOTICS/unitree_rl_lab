"""probe_bimanual.py — Bimanual Engagement Ratio (BER) acceptance probe.

Measures the two-gate acceptance criterion locked 2026-06-13:
  Gate 1 (Accuracy):  SR  ≥ 0.95  (success = pressure_success_hold, hold_steps=50)
  Gate 2 (Bimanual):  %{episodes : BER ≥ 0.85} ≥ 0.85

where BER per episode = (#steps bimanual) / (#steps in contact-window),
contact-window = first-contact step → episode end (success or timeout).

Interaction is defined via TORQUE CONTRIBUTION (not raw force magnitude).
Per hand h per step:
  p_h   = palm body world position (left_hand_base_link / right_hand_base_link)
  F_h   = net_forces_w from palm ContactSensor (PRIMARY path — R1 skipped,
           ContactSensorCfg with filter_prim_paths_expr on articulation links
           always returns 0; see ADR 0007 and docs/agents/sim-isaaclab.md:481).
           net_forces_w ≈ valve contact force because during mid-stroke turning
           the palm contacts are overwhelmingly valve rim (palm contacts other
           surfaces only transiently at episode start before contact window opens).
  p_v   = valve hub body_pos_w (hub body = "mesh_50_AL_250_B7_8_A_stl", body_ids[0])
  â     = hinge axis in world frame = (0, 0, 1)  [world-Z, confirmed: valve spawns
           at rot=(0.707,0,0,0.707) = +90° Z; RevoluteJoint local axis = Z;
           after +90° Z rotation, local Z → world Z → hinge axis stays world-Z]
  r_h   = p_h − p_v            (moment arm vector)
  F_wheel,h = −F_h             (Newton 3rd law: net_forces_w = force ON palm FROM env)
  τ_h   = (r_h × F_wheel,h) · â    (torque about hinge, scalar)
  turn_dir = sign(p_des − p_now)   (g(θ) monotone ↑, CCW opens → sign of required turn)
  contributes_h = (sign(τ_h) == turn_dir) AND (|τ_h| >= tau_min)
  bimanual_step = contributes_L AND contributes_R
  in_contact_L = |F_L| >= contact_thresh  (N), similarly R
  in_contact_h  = in_contact_L OR in_contact_R  (contact-window entry)

Contact-window opens on the first step where either hand force >= contact_thresh (0.5 N)
and stays open until end-of-episode (no re-close on transient drop).

Also reports dominance breakdown per episode:
  L-only  : contributes_L AND NOT contributes_R
  R-only  : contributes_R AND NOT contributes_L
  both    : bimanual
  none    : neither

Per-episode CSV columns (--out_csv):
  ep_idx, success, ep_len, contact_window_steps, ber, l_only_frac,
  r_only_frac, both_frac, none_frac, p_des, theta_init

Shared eval infra for:  R4 (policy selection), B2 (ADR-0009 SR parity),
                         E3 (sparse-feedback sweep), Cap.Resultados numbers.

Usage (from /home/tr-robotics/unitree_rl_lab/scripts/rsl_rl/):
    conda activate env_isaaclab
    python probe_bimanual.py \\
        --task Unitree-G1-29dof-ValveTurn-v5 \\
        --checkpoint /path/to/model.pt \\
        --num_envs 64 \\
        --num_episodes 200 \\
        --out_csv probe_results.csv \\
        --headless

Do NOT exceed --num_envs 128 on RTX 5080 16GB (physics/contact buffers OOM at 4096).
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import importlib

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Bimanual Engagement Ratio (BER) acceptance probe.")
parser.add_argument("--task", type=str, default="Unitree-G1-29dof-ValveTurn-v5",
                    help="Gym task ID (e.g. Unitree-G1-29dof-ValveTurn-v5).")
parser.add_argument("--checkpoint", type=str, required=True,
                    help="Path to RSL-RL .pt checkpoint.")
parser.add_argument("--num_envs", type=int, default=64,
                    help="Parallel envs. Keep <=128 on RTX 5080 16 GB to avoid OOM.")
parser.add_argument("--num_episodes", type=int, default=200,
                    help="Total episodes to collect (across all envs). Minimum 200.")
parser.add_argument("--out_csv", type=str, default="probe_bimanual_results.csv",
                    help="Path to per-episode CSV output.")
parser.add_argument("--tau_min", type=float, default=0.1,
                    help="Minimum |torque| (N·m) for a hand to count as contributing.")
parser.add_argument("--contact_thresh", type=float, default=0.5,
                    help="Force magnitude (N) above which a palm is considered in contact.")
# RSL-RL + AppLauncher flags (mirrors play.py / probe_reset_direction.py)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import csv
import os

import torch
import gymnasium as gym
from rsl_rl.runners import OnPolicyRunner

importlib.import_module("isaaclab_tasks")
importlib.import_module("unitree_rl_lab.tasks")

from isaaclab.assets import Articulation
from isaaclab.sensors import ContactSensor
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from isaaclab_tasks.utils import get_checkpoint_path

import unitree_rl_lab.tasks.manipulation.mdp.pressure as pressure_mdp
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_HINGE_AXIS = torch.tensor([0.0, 0.0, 1.0])   # world-Z; see module docstring


def _get_palm_force(sensor: ContactSensor) -> torch.Tensor:
    """Net force on the palm body in world frame (N). Shape: (num_envs, 3).

    net_forces_w shape is (num_envs, 1, 3) for single-body sensors.
    ContactSensorCfg has no filter_prim_paths_expr (ADR 0007 / sim-isaaclab.md:481
    documents that filter on articulation links always returns 0 — R1 skipped).
    The unfiltered net_forces_w approximates valve contact force; during the
    contact window palm contacts are overwhelmingly valve rim.
    """
    return sensor.data.net_forces_w[:, 0, :]   # (num_envs, 3)


def _torque_contribution(
    palm_pos_w: torch.Tensor,      # (num_envs, 3)
    hub_pos_w: torch.Tensor,       # (num_envs, 3)
    palm_force_env: torch.Tensor,  # (num_envs, 3)  — force ON palm FROM env
    hinge_axis: torch.Tensor,      # (3,)
) -> torch.Tensor:
    """Scalar torque about hinge for one hand. Shape: (num_envs,).

    τ = (r × F_wheel) · â
    where r = p_h − p_v  (moment arm)
          F_wheel = −F_palm  (Newton 3rd: env pushes palm → palm pushes wheel opposite)
    """
    r = palm_pos_w - hub_pos_w                     # (num_envs, 3)
    F_wheel = -palm_force_env                       # (num_envs, 3)
    cross = torch.linalg.cross(r, F_wheel, dim=-1)  # (num_envs, 3)
    tau = (cross * hinge_axis).sum(dim=-1)          # (num_envs,)
    return tau


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    print(f"\n{'='*70}")
    print("probe_bimanual.py — Bimanual Engagement Ratio (BER) acceptance probe")
    print(f"Task:        {args_cli.task}")
    print(f"Checkpoint:  {args_cli.checkpoint}")
    print(f"num_envs:    {args_cli.num_envs}   num_episodes: {args_cli.num_episodes}")
    print(f"tau_min:     {args_cli.tau_min} N·m   contact_thresh: {args_cli.contact_thresh} N")
    print(f"out_csv:     {args_cli.out_csv}")
    print(f"{'='*70}\n")

    # ------------------------------------------------------------------ env --
    env_cfg = parse_env_cfg(
        args_cli.task,
        device="cuda:0",
        num_envs=args_cli.num_envs,
        use_fabric=True,
        entry_point_key="env_cfg_entry_point",
    )
    env_cfg.scene.num_envs = args_cli.num_envs

    raw_env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(raw_env, clip_actions=True)

    # ---------------------------------------------------------------- policy --
    resume_path = retrieve_file_path(args_cli.checkpoint)
    # cli_args / agent_cfg not available here (no --experiment_name flag on this probe);
    # build a minimal agent cfg by querying the registry (mirrors play.py pattern).
    import cli_args as rsl_cli   # local to scripts/rsl_rl/
    import sys
    # Inject a minimal Namespace to satisfy parse_rsl_rl_cfg (load_run / load_checkpoint).
    import types
    _fake_args = types.SimpleNamespace(
        task=args_cli.task,
        checkpoint=args_cli.checkpoint,
        load_run=None,
        load_checkpoint=None,
        num_envs=args_cli.num_envs,
        device="cuda:0",
    )
    agent_cfg = rsl_cli.parse_rsl_rl_cfg(args_cli.task, _fake_args)
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, "5.0.1")

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # ---------------------------------------------------------- scene handles --
    base_env = raw_env.unwrapped   # ManagerBasedRLEnv

    # valve_rig Articulation — scene key confirmed base_cfg.py ValveSceneCfg
    valve_art: Articulation = base_env.scene["valve_rig"]
    # robot Articulation — scene key confirmed base_cfg.py ValveSceneCfg
    robot_art: Articulation = base_env.scene["robot"]
    # ContactSensors — scene keys confirmed base_cfg.py ValveSceneCfg
    left_sensor: ContactSensor = base_env.scene["left_palm_sensor"]
    right_sensor: ContactSensor = base_env.scene["right_palm_sensor"]

    # Hub body index for body_pos_w — confirmed base_cfg.py _HUB_BODY_NAME
    # SceneEntityCfg resolution: body_ids are resolved during env setup.
    # We look up the body index ourselves to avoid re-resolving SceneEntityCfg.
    hub_body_name = "mesh_50_AL_250_B7_8_A_stl"
    body_names_list = valve_art.data.body_names
    if hub_body_name not in body_names_list:
        raise RuntimeError(
            f"Hub body '{hub_body_name}' not found in valve_rig bodies: {body_names_list}"
        )
    hub_body_idx = body_names_list.index(hub_body_name)

    # palm body index for left/right robot articulation
    # left_hand_base_link / right_hand_base_link — confirmed base_cfg.py sensors
    robot_body_names = robot_art.data.body_names
    for _name in ("left_hand_base_link", "right_hand_base_link"):
        if _name not in robot_body_names:
            raise RuntimeError(
                f"Palm body '{_name}' not found in robot bodies."
            )
    left_palm_idx  = robot_body_names.index("left_hand_base_link")
    right_palm_idx = robot_body_names.index("right_hand_base_link")

    dev = base_env.device
    hinge_axis = _HINGE_AXIS.to(dev)

    # ------------------------------------------------- per-episode accumulators --
    # We collect num_episodes completed episodes (any env).
    num_envs = args_cli.num_envs
    tau_min = args_cli.tau_min
    contact_thresh = args_cli.contact_thresh
    num_episodes_target = args_cli.num_episodes

    # Per-env in-flight state (reset on episode boundary)
    contact_window_open  = torch.zeros(num_envs, dtype=torch.bool, device=dev)
    contact_window_steps = torch.zeros(num_envs, dtype=torch.long, device=dev)
    bimanual_steps       = torch.zeros(num_envs, dtype=torch.long, device=dev)
    l_only_steps         = torch.zeros(num_envs, dtype=torch.long, device=dev)
    r_only_steps         = torch.zeros(num_envs, dtype=torch.long, device=dev)
    none_steps           = torch.zeros(num_envs, dtype=torch.long, device=dev)

    # Snapshot of initial θ per env (captured after each reset)
    theta_init_snap = torch.zeros(num_envs, dtype=torch.float32, device=dev)
    # p_des snapshot
    p_des_snap      = torch.zeros(num_envs, dtype=torch.float32, device=dev)

    # Completed-episode records
    episode_records: list[dict] = []

    # ----------------------------------------------------------------- loop --
    obs, _ = env.get_observations()
    # Capture initial snapshots (post-first-reset)
    theta_init_snap[:] = valve_art.data.joint_pos[:, 0]
    p_des_buf = pressure_mdp.p_des(base_env)
    if p_des_buf is not None:
        p_des_snap[:] = p_des_buf.reshape(num_envs)

    ep_count = 0

    while simulation_app.is_running() and ep_count < num_episodes_target:
        with torch.inference_mode():
            actions = policy(obs)
            # RslRlVecEnvWrapper.step() returns (obs, rew, terminated, info)
            # confirmed via play.py line 170: obs, _, _, _ = env.step(actions)
            obs, _rew, terminated, _info = env.step(actions)

        # ------------------------------------------------ per-step BER update --
        # Valve hub world position
        hub_pos = valve_art.data.body_pos_w[:, hub_body_idx, :]  # (N, 3)

        # Palm world positions
        left_pos  = robot_art.data.body_pos_w[:, left_palm_idx,  :]  # (N, 3)
        right_pos = robot_art.data.body_pos_w[:, right_palm_idx, :]  # (N, 3)

        # Palm forces from unfiltered ContactSensors (see module docstring / ADR 0007)
        F_left  = _get_palm_force(left_sensor)   # (N, 3)
        F_right = _get_palm_force(right_sensor)  # (N, 3)

        # Force magnitudes for contact-window gating
        f_left_mag  = F_left.norm(dim=-1)   # (N,)
        f_right_mag = F_right.norm(dim=-1)  # (N,)
        either_contact = (f_left_mag >= contact_thresh) | (f_right_mag >= contact_thresh)

        # Open contact window (latching — never closes mid-episode)
        contact_window_open = contact_window_open | either_contact

        # Torque contributions
        tau_L = _torque_contribution(left_pos,  hub_pos, F_left,  hinge_axis)  # (N,)
        tau_R = _torque_contribution(right_pos, hub_pos, F_right, hinge_axis)  # (N,)

        # Turn direction = sign(p_des − p_now); g(θ) monotone ↑ so CCW = positive
        p_now_vals = pressure_mdp.p_now(base_env)         # (N,)
        p_des_vals = pressure_mdp.p_des(base_env)         # (N,) or None
        if p_des_vals is None:
            turn_dir = torch.ones(num_envs, device=dev)
        else:
            turn_dir = torch.sign(p_des_vals - p_now_vals)  # (N,)

        tau_sign_L  = torch.sign(tau_L)
        tau_sign_R  = torch.sign(tau_R)
        contrib_L   = (tau_sign_L == turn_dir) & (tau_L.abs() >= tau_min)   # (N,)
        contrib_R   = (tau_sign_R == turn_dir) & (tau_R.abs() >= tau_min)   # (N,)

        bimanual_step = contrib_L & contrib_R             # (N,)
        l_only_step   = contrib_L & ~contrib_R
        r_only_step   = contrib_R & ~contrib_L
        none_step     = ~contrib_L & ~contrib_R

        # Accumulate only within the open contact window
        in_window = contact_window_open
        contact_window_steps += in_window.long()
        bimanual_steps       += (in_window & bimanual_step).long()
        l_only_steps         += (in_window & l_only_step).long()
        r_only_steps         += (in_window & r_only_step).long()
        none_steps           += (in_window & none_step).long()

        # --------------------------------------------- episode completion --
        # terminated is a bool tensor (num_envs,) from RslRlVecEnvWrapper
        done_mask = terminated.bool()

        # Check success via _success_hold_counter — confirmed terminations.py:84
        # _success_hold_counter >= hold_steps → True on the step it fires.
        hold_counter = getattr(base_env, "_success_hold_counter", None)

        done_idxs = done_mask.nonzero(as_tuple=False).squeeze(-1)
        for idx in done_idxs:
            i = int(idx.item())
            ep_len = int(base_env.episode_length_buf[i].item())
            win_steps = int(contact_window_steps[i].item())
            bim_steps = int(bimanual_steps[i].item())
            lo_steps  = int(l_only_steps[i].item())
            ro_steps  = int(r_only_steps[i].item())
            no_steps  = int(none_steps[i].item())

            ber = bim_steps / win_steps if win_steps > 0 else 0.0
            lo_frac  = lo_steps  / win_steps if win_steps > 0 else 0.0
            ro_frac  = ro_steps  / win_steps if win_steps > 0 else 0.0
            no_frac  = no_steps  / win_steps if win_steps > 0 else 0.0
            both_frac = bim_steps / win_steps if win_steps > 0 else 0.0

            # Success flag from hold counter
            if hold_counter is not None:
                success = bool((hold_counter[i] >= 50).item())
            else:
                success = False

            episode_records.append({
                "ep_idx":              len(episode_records),
                "success":             int(success),
                "ep_len":              ep_len,
                "contact_window_steps": win_steps,
                "ber":                 ber,
                "l_only_frac":         lo_frac,
                "r_only_frac":         ro_frac,
                "both_frac":           both_frac,
                "none_frac":           no_frac,
                "p_des":               float(p_des_snap[i].item()),
                "theta_init":          float(theta_init_snap[i].item()),
            })
            ep_count += 1

            # Reset per-env accumulators for this env slot
            contact_window_open[i]  = False
            contact_window_steps[i] = 0
            bimanual_steps[i]       = 0
            l_only_steps[i]         = 0
            r_only_steps[i]         = 0
            none_steps[i]           = 0

            if ep_count >= num_episodes_target:
                break

        # Snapshot θ_init + p_des for envs that just reset (next episode)
        if done_mask.any():
            # After env.step, done envs have been auto-reset by the VecEnv wrapper.
            theta_init_snap[done_mask] = valve_art.data.joint_pos[:, 0][done_mask]
            p_des_now = pressure_mdp.p_des(base_env)
            if p_des_now is not None:
                p_des_snap[done_mask] = p_des_now.reshape(num_envs)[done_mask]

        if ep_count >= num_episodes_target:
            break

    # ----------------------------------------------------------- summary --
    n = len(episode_records)
    if n == 0:
        print("[WARN] No episodes completed.")
        env.close()
        return 1

    successes = [r["success"] for r in episode_records]
    bers      = [r["ber"]     for r in episode_records]

    sr  = sum(successes) / n
    pct_ber_pass = sum(1 for b in bers if b >= 0.85) / n
    mean_ber   = sum(bers) / n
    median_ber = float(torch.tensor(bers).median().item())

    gate1_pass = sr  >= 0.95
    gate2_pass = pct_ber_pass >= 0.85

    print(f"\n{'='*70}")
    print(f"RESULTS  (n={n} episodes, tau_min={args_cli.tau_min} N·m)")
    print(f"{'='*70}")
    print(f"  SR                     : {sr:.4f}   gate ≥0.95 → {'PASS' if gate1_pass else 'FAIL'}")
    print(f"  %eps(BER≥0.85)         : {pct_ber_pass:.4f}   gate ≥0.85 → {'PASS' if gate2_pass else 'FAIL'}")
    print(f"  mean BER               : {mean_ber:.4f}")
    print(f"  median BER             : {median_ber:.4f}")
    print()

    # Dominance breakdown (per episode, mean fraction)
    def _mean_frac(key: str) -> float:
        return sum(r[key] for r in episode_records) / n

    print(f"  Dominance (mean frac over contact window):")
    print(f"    both  (bimanual)     : {_mean_frac('both_frac'):.4f}")
    print(f"    L-only               : {_mean_frac('l_only_frac'):.4f}")
    print(f"    R-only               : {_mean_frac('r_only_frac'):.4f}")
    print(f"    none                 : {_mean_frac('none_frac'):.4f}")
    print()

    verdict = "PASS" if (gate1_pass and gate2_pass) else "FAIL"
    print(f"  OVERALL: {verdict}  ({'both gates' if gate1_pass and gate2_pass else 'gate(s) failed'})")
    print(f"{'='*70}\n")

    # --------------------------------------------------------- CSV output --
    csv_path = os.path.abspath(args_cli.out_csv)
    fieldnames = [
        "ep_idx", "success", "ep_len", "contact_window_steps", "ber",
        "l_only_frac", "r_only_frac", "both_frac", "none_frac", "p_des", "theta_init",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(episode_records)
    print(f"[INFO] Per-episode CSV written to: {csv_path}")

    env.close()
    return 0 if (gate1_pass and gate2_pass) else 1


if __name__ == "__main__":
    rc = main()
    simulation_app.close()
    raise SystemExit(rc)
