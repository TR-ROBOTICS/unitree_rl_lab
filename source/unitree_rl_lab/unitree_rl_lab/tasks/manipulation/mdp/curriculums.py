"""Curriculum terms for valve-turn task.

  rim_distance_weight_anneal — anneal `rim_distance` reward weight once policy
    demonstrates consistent hand-to-rim proximity.

  turn_auto_curriculum_stage — 3-stage auto-curriculum (v4a): expands θ then p_des.
  turn_auto_curriculum_stage_easy — 4-stage auto-curriculum (v4ae): adds dataset arm init.

  turn_smooth_curriculum_v5 — v5 smooth dual-axis curriculum: expands θ and p_des
    simultaneously in small steps, then mixes in dataset arm init gradually.
"""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def rim_distance_weight_anneal(
    env: "ManagerBasedRLEnv",
    env_ids: Sequence[int],
    reward_term_name: str = "rim_distance",
    threshold_frac: float = 0.8,
    decay: float = 0.98,
    floor: float = 0.05,
) -> torch.Tensor:
    """Multiplicative anneal of `rim_distance` reward weight.

    Pattern mirrors `locomotion.mdp.curriculums.lin_vel_cmd_levels`: fires once
    per `max_episode_length` global steps. When the mean per-step raw reward of
    `reward_term_name` exceeds `threshold_frac · current_weight` (i.e. policy
    earns ≥80% of theoretical maximum), shrink the weight by `decay` (default
    0.98, half-life ≈ 34 gated episodes). Floor at `floor` so the shaping signal
    never fully disappears.

    Rationale: high initial weight (e.g. 2.0) forces the policy to learn rim
    contact early; once contact is consistent, decaying the weight lets
    `pressure_progress` dominate so the policy stops static-gripping and starts
    turning the wheel. Multiplicative decay is self-scaling — big drops when
    weight high, gentle drops near floor — matches log-scale reward weight
    intuition.

    Args:
        reward_term_name: Reward term whose weight is annealed. Default
                          "rim_distance".
        threshold_frac:   Mean per-step raw reward as fraction of current weight
                          required to trigger a decay step. Default 0.8.
        decay:            Multiplicative factor applied to weight on trigger.
                          Default 0.98.
        floor:            Minimum weight value — anneal clamps here. Default 0.05.

    Returns:
        Scalar tensor — current weight (for logging).
    """
    rt = env.reward_manager.get_term_cfg(reward_term_name)
    # IsaacLab's RewardManager._episode_sums[name] accumulates weight·raw per
    # step WITHOUT dt (dt is applied only to the total reward buffer returned
    # to RSL-RL). Dividing by max_episode_length (steps, int) gives mean
    # per-step weighted reward; dividing again by weight gives raw fraction
    # ∈ (0, 1] for rim_distance_reward.
    #
    # Earlier bug: divided by max_episode_length_s (= num_steps · dt). Because
    # _episode_sums has no dt, this overcounted by 1/dt ≈ 50 → raw_frac ≈ 1.65
    # for typical distant-hand reward → curriculum always fired immediately.
    weighted_per_step = (
        torch.mean(env.reward_manager._episode_sums[reward_term_name][env_ids])
        / env.max_episode_length
    )
    raw_frac = weighted_per_step / max(rt.weight, 1e-8)

    if env.common_step_counter % env.max_episode_length == 0:
        if raw_frac > threshold_frac and rt.weight > floor:
            rt.weight = max(floor, rt.weight * decay)

    return torch.tensor(rt.weight, device=env.device)


def turn_auto_curriculum_stage(
    env: "ManagerBasedRLEnv",
    env_ids: Sequence[int],
    success_threshold: float = 0.85,
    window_iters: int = 100,
    num_steps_per_env: int = 24,
    theta_max: float = 50.27,
    p_min_target: float = 15.0,
    p_max_target: float = 200.0,
) -> torch.Tensor:
    """Auto-curriculum for valve-turn (v4auto): expands θ_init then p_des range.

    Mirrors the manual v0→v1→v2 curriculum progression automatically within a
    single training run, using rolling success rate as the advancement trigger.

    Stage 0 (start): θ_init = θ_min (fixed), p_des = 50 PSI (fixed) — same as v0.
    Stage 1:          θ_init ∈ [θ_min, θ_max] (random)               — same as v1.
                      Triggered: rolling SR ≥ success_threshold over window_iters iters.
    Stage 2:          p_des ∈ [p_min_target, p_max_target] (random)   — same as v2.
                      Triggered: rolling SR ≥ success_threshold again.

    Implementation:
      Mutates EventTermCfg params in-place via env.event_manager.get_term_cfg().
      Curriculum fires in _reset_idx() BEFORE event_manager.apply(), so updated
      params take effect in the same reset cycle (no lag by one episode).

    Window is iter-based (not episode-count): evaluation fires every
    window_iters × num_steps_per_env policy steps, matching training iteration
    cadence regardless of episode length or num_envs.

    Tracking state stored on env as:
      _autocurr_stage: int (0/1/2)
      _autocurr_window_done: int
      _autocurr_window_success: int
      _autocurr_last_eval_step: int

    Args:
        success_threshold: Rolling success rate required to advance stage.
                           Default 0.85 (matches v0-v4 gate criterion).
        window_iters:      Training iterations per evaluation window.
                           Default 100 (≈ 2400 policy steps at num_steps_per_env=24).
        num_steps_per_env: RSL-RL OnPolicyRunner.num_steps_per_env. Default 24.
        theta_max:         Full θ_init upper bound (rad). Default 50.27 (8 rev).
        p_min_target:      Full p_des lower bound (PSI). Default 15.0.
        p_max_target:      Full p_des upper bound (PSI). Default 200.0.

    Returns:
        Scalar tensor — current curriculum stage (0/1/2), for TensorBoard logging.
    """
    if len(env_ids) == 0:
        stage = getattr(env, "_autocurr_stage", 0)
        return torch.tensor(float(stage), device=env.device)

    # Initialise tracking state once
    if not hasattr(env, "_autocurr_stage"):
        env._autocurr_stage = 0
        env._autocurr_window_done = 0
        env._autocurr_window_success = 0
        env._autocurr_last_eval_step = 0

    # env.reset_terminated is set in the step loop before _reset_idx; on the very
    # first env.reset() call it does not exist yet — treat as 0 successes.
    reset_terminated = getattr(env, "reset_terminated", None)
    n_success = int(reset_terminated[env_ids].sum().item()) if reset_terminated is not None else 0
    n_done = len(env_ids)

    env._autocurr_window_done += n_done
    env._autocurr_window_success += n_success

    # Evaluate once per window_iters training iterations using common_step_counter.
    # common_step_counter increments by 1 per env.step() call (= 1 policy step).
    # RSL-RL calls env.step() num_steps_per_env times per training iteration.
    window_step_size = window_iters * num_steps_per_env
    steps_since_eval = env.common_step_counter - env._autocurr_last_eval_step

    if steps_since_eval >= window_step_size and env._autocurr_stage < 2:
        success_rate = env._autocurr_window_success / max(env._autocurr_window_done, 1)

        if success_rate >= success_threshold:
            if env._autocurr_stage == 0:
                # Stage 0 → 1: unlock random θ_init
                cfg = env.event_manager.get_term_cfg("reset_valve_angle")
                cfg.params["angle_max"] = theta_max
                env._autocurr_stage = 1
                print(
                    f"[AutoCurr] Stage 0→1 | SR={success_rate:.3f} ≥ {success_threshold} "
                    f"over {env._autocurr_window_done} eps "
                    f"(step {env.common_step_counter}) | "
                    f"angle_max → {theta_max:.2f} rad"
                )
            elif env._autocurr_stage == 1:
                # Stage 1 → 2: unlock random p_des
                cfg = env.event_manager.get_term_cfg("reset_p_des")
                cfg.params["p_min"] = p_min_target
                cfg.params["p_max"] = p_max_target
                env._autocurr_stage = 2
                print(
                    f"[AutoCurr] Stage 1→2 | SR={success_rate:.3f} ≥ {success_threshold} "
                    f"over {env._autocurr_window_done} eps "
                    f"(step {env.common_step_counter}) | "
                    f"p_des → [{p_min_target:.1f}, {p_max_target:.1f}] PSI"
                )
        else:
            print(
                f"[AutoCurr] Stage {env._autocurr_stage} hold | "
                f"SR={success_rate:.3f} < {success_threshold} "
                f"over {env._autocurr_window_done} eps (step {env.common_step_counter})"
            )

        # Reset window and advance eval checkpoint regardless of outcome
        env._autocurr_window_done = 0
        env._autocurr_window_success = 0
        env._autocurr_last_eval_step = env.common_step_counter

    return torch.tensor(float(env._autocurr_stage), device=env.device)


def turn_auto_curriculum_stage_easy(
    env: "ManagerBasedRLEnv",
    env_ids: Sequence[int],
    success_threshold: float = 0.85,
    window_iters: int = 100,
    num_steps_per_env: int = 24,
    theta_max: float = 50.27,
    p_min_target: float = 15.0,
    p_max_target: float = 200.0,
) -> torch.Tensor:
    """4-stage auto-curriculum (v4autoe): mirrors human-guided v0→v1→v2→v3 chain.

    Stage 0 (start):  θ_init = θ_min (fixed), p_des = 50 PSI, pre-grip arm init ← v0
    Stage 1:          θ_init ∈ [θ_min, θ_max] (random),       p_des = 50 PSI    ← v1
    Stage 2:          θ_init random, p_des ∈ [p_min, p_max]   (random)           ← v2
    Stage 3:          θ_init random, p_des random, dataset arm init               ← v3+

    Stage 2→3 sets env._autocurr_use_dataset = True, which reset_arm_staged reads.

    Window is iter-based (not episode-count): evaluation fires every
    window_iters × num_steps_per_env policy steps.

    Tracking state on env:
      _autocurr_stage: int (0/1/2/3)
      _autocurr_window_done: int
      _autocurr_window_success: int
      _autocurr_last_eval_step: int
      _autocurr_use_dataset: bool (False until Stage 3)

    Args:
        success_threshold: SR to advance stage. Default 0.85.
        window_iters:      Training iterations per evaluation window. Default 100.
        num_steps_per_env: RSL-RL OnPolicyRunner.num_steps_per_env. Default 24.
        theta_max:         Full θ_init upper bound (rad). Default 50.27.
        p_min_target:      Full p_des lower bound (PSI). Default 15.0.
        p_max_target:      Full p_des upper bound (PSI). Default 200.0.

    Returns:
        Scalar tensor — current stage (0/1/2/3), for TensorBoard logging.
    """
    if len(env_ids) == 0:
        stage = getattr(env, "_autocurr_stage", 0)
        return torch.tensor(float(stage), device=env.device)

    if not hasattr(env, "_autocurr_stage"):
        env._autocurr_stage = 0
        env._autocurr_window_done = 0
        env._autocurr_window_success = 0
        env._autocurr_last_eval_step = 0
        env._autocurr_use_dataset = False

    reset_terminated = getattr(env, "reset_terminated", None)
    n_success = int(reset_terminated[env_ids].sum().item()) if reset_terminated is not None else 0
    n_done = len(env_ids)

    env._autocurr_window_done += n_done
    env._autocurr_window_success += n_success

    # Evaluate once per window_iters training iterations
    window_step_size = window_iters * num_steps_per_env
    steps_since_eval = env.common_step_counter - env._autocurr_last_eval_step

    if steps_since_eval >= window_step_size and env._autocurr_stage < 3:
        success_rate = env._autocurr_window_success / max(env._autocurr_window_done, 1)

        if success_rate >= success_threshold:
            if env._autocurr_stage == 0:
                cfg = env.event_manager.get_term_cfg("reset_valve_angle")
                cfg.params["angle_max"] = theta_max
                env._autocurr_stage = 1
                print(
                    f"[AutoCurrE] Stage 0→1 | SR={success_rate:.3f} ≥ {success_threshold} "
                    f"over {env._autocurr_window_done} eps "
                    f"(step {env.common_step_counter}) | "
                    f"angle_max → {theta_max:.2f} rad"
                )
            elif env._autocurr_stage == 1:
                cfg = env.event_manager.get_term_cfg("reset_p_des")
                cfg.params["p_min"] = p_min_target
                cfg.params["p_max"] = p_max_target
                env._autocurr_stage = 2
                print(
                    f"[AutoCurrE] Stage 1→2 | SR={success_rate:.3f} ≥ {success_threshold} "
                    f"over {env._autocurr_window_done} eps "
                    f"(step {env.common_step_counter}) | "
                    f"p_des → [{p_min_target:.1f}, {p_max_target:.1f}] PSI"
                )
            elif env._autocurr_stage == 2:
                env._autocurr_use_dataset = True
                env._autocurr_stage = 3
                print(
                    f"[AutoCurrE] Stage 2→3 | SR={success_rate:.3f} ≥ {success_threshold} "
                    f"over {env._autocurr_window_done} eps "
                    f"(step {env.common_step_counter}) | "
                    f"Unlocking dataset arm init"
                )
        else:
            print(
                f"[AutoCurrE] Stage {env._autocurr_stage} hold | "
                f"SR={success_rate:.3f} < {success_threshold} "
                f"over {env._autocurr_window_done} eps (step {env.common_step_counter})"
            )

        env._autocurr_window_done = 0
        env._autocurr_window_success = 0
        env._autocurr_last_eval_step = env.common_step_counter

    return torch.tensor(float(env._autocurr_stage), device=env.device)


def turn_smooth_curriculum_v5(
    env: "ManagerBasedRLEnv",
    env_ids: Sequence[int],
    success_threshold: float = 0.85,
    window_iters: int = 100,
    num_steps_per_env: int = 24,
    theta_min: float = 9.42,
    theta_max: float = 50.27,
    theta_mid: float = 29.75,
    theta_step: float = 2.04,
    p_min: float = 15.0,
    p_max: float = 200.0,
    p_mid: float = 107.0,
    p_step: float = 9.25,
    dataset_step: float = 0.10,
) -> torch.Tensor:
    """v5 smooth dual-axis curriculum.

    Stage 0 — simultaneous θ and p_des range expansion:
      Start: θ_init ∈ [θ_mid−θ_step, θ_mid+θ_step], p_des ∈ [p_mid, p_mid]
      Each 85% SR trigger: θ lo/hi ±θ_step, p_des lo/hi ±p_step (5% of span each)
      Advance to Stage 1 when both ranges fully open.

    Stage 1 — dataset arm init mixing:
      Each 85% SR trigger: dataset_pct += dataset_step (+10%)
      Advance to Stage 2 when dataset_pct >= 1.0.

    Stage 2 — fully open, 100% dataset init. Terminal.

    Mutates event cfg params in-place each trigger:
      reset_valve_angle: angle_min, angle_max
      reset_p_des:       p_min, p_max
    Sets env._v5curr_dataset_pct for reset_arm_v5 event to read.
    """
    if len(env_ids) == 0:
        stage = getattr(env, "_v5curr_stage", 0)
        return torch.tensor(float(stage), device=env.device)

    if not hasattr(env, "_v5curr_stage"):
        env._v5curr_stage = 0
        env._v5curr_theta_lo = theta_mid - theta_step
        env._v5curr_theta_hi = theta_mid + theta_step
        env._v5curr_p_lo = p_mid
        env._v5curr_p_hi = p_mid
        env._v5curr_dataset_pct = 0.0
        env._v5curr_window_done = 0
        env._v5curr_window_success = 0
        env._v5curr_last_eval_step = 0
        _v5_apply_theta(env, env._v5curr_theta_lo, env._v5curr_theta_hi)
        _v5_apply_p(env, env._v5curr_p_lo, env._v5curr_p_hi)

    reset_terminated = getattr(env, "reset_terminated", None)
    n_success = int(reset_terminated[env_ids].sum().item()) if reset_terminated is not None else 0
    env._v5curr_window_done += len(env_ids)
    env._v5curr_window_success += n_success

    window_step_size = window_iters * num_steps_per_env
    steps_since_eval = env.common_step_counter - env._v5curr_last_eval_step

    if steps_since_eval >= window_step_size and env._v5curr_stage < 2:
        sr = env._v5curr_window_success / max(env._v5curr_window_done, 1)

        if sr >= success_threshold:
            if env._v5curr_stage == 0:
                env._v5curr_theta_lo = max(theta_min, env._v5curr_theta_lo - theta_step)
                env._v5curr_theta_hi = min(theta_max, env._v5curr_theta_hi + theta_step)
                env._v5curr_p_lo = max(p_min, env._v5curr_p_lo - p_step)
                env._v5curr_p_hi = min(p_max, env._v5curr_p_hi + p_step)
                _v5_apply_theta(env, env._v5curr_theta_lo, env._v5curr_theta_hi)
                _v5_apply_p(env, env._v5curr_p_lo, env._v5curr_p_hi)

                theta_full = (env._v5curr_theta_lo <= theta_min) and (env._v5curr_theta_hi >= theta_max)
                p_full = (env._v5curr_p_lo <= p_min) and (env._v5curr_p_hi >= p_max)

                print(
                    f"[V5Curr] Stage 0 expand | SR={sr:.3f} | "
                    f"θ=[{env._v5curr_theta_lo:.2f},{env._v5curr_theta_hi:.2f}] "
                    f"p=[{env._v5curr_p_lo:.1f},{env._v5curr_p_hi:.1f}] "
                    f"(step {env.common_step_counter})"
                )
                if theta_full and p_full:
                    env._v5curr_stage = 1
                    print(f"[V5Curr] Stage 0→1 | fully open (step {env.common_step_counter})")

            elif env._v5curr_stage == 1:
                env._v5curr_dataset_pct = min(1.0, env._v5curr_dataset_pct + dataset_step)
                print(
                    f"[V5Curr] Stage 1 expand | SR={sr:.3f} | "
                    f"dataset_pct={env._v5curr_dataset_pct:.1%} (step {env.common_step_counter})"
                )
                if env._v5curr_dataset_pct >= 1.0:
                    env._v5curr_stage = 2
                    print(f"[V5Curr] Stage 1→2 | 100% dataset init (step {env.common_step_counter})")
        else:
            print(
                f"[V5Curr] Stage {env._v5curr_stage} hold | "
                f"SR={sr:.3f} < {success_threshold} (step {env.common_step_counter})"
            )

        env._v5curr_window_done = 0
        env._v5curr_window_success = 0
        env._v5curr_last_eval_step = env.common_step_counter

    return torch.tensor(float(env._v5curr_stage), device=env.device)


def _v5_apply_theta(env: "ManagerBasedRLEnv", lo: float, hi: float) -> None:
    cfg = env.event_manager.get_term_cfg("reset_valve_angle")
    cfg.params["angle_min"] = lo
    cfg.params["angle_max"] = hi


def _v5_apply_p(env: "ManagerBasedRLEnv", lo: float, hi: float) -> None:
    cfg = env.event_manager.get_term_cfg("reset_p_des")
    cfg.params["p_min"] = lo
    cfg.params["p_max"] = hi
