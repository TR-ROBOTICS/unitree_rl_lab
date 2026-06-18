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
    theta_start_lo: float | None = None,
    theta_start_hi: float | None = None,
    p_start: float | None = None,
) -> torch.Tensor:
    """v5 smooth dual-axis curriculum.

    Stage 0 — simultaneous θ and p_des range expansion:
      Start: θ_init ∈ [theta_start_lo, theta_start_hi], p_des ∈ [p_start, p_start]
             Defaults: [theta_min, theta_min+theta_step], p_start=50.0 (v0-style)
      Each 85% SR trigger: θ_hi += θ_step (lo clamps at theta_min), p_des ±p_step
      Advance to Stage 1 when θ_hi >= theta_max AND p fully open.

    Stage 1 — dataset arm init mixing:
      Each 85% SR trigger: dataset_pct += dataset_step (+10%)
      Advance to Stage 2 when dataset_pct >= 1.0.

    Stage 2 — fully open, 100% dataset init. Terminal.

    Mutates event cfg params in-place each trigger:
      reset_valve_angle: angle_min, angle_max
      reset_p_des:       p_min, p_max
    Syncs env._curr_dataset_pct (canonical mixing fraction) for the unified
    reset_arm_mixed event to read.
    """
    if theta_start_lo is None:
        theta_start_lo = theta_min
    if theta_start_hi is None:
        theta_start_hi = theta_min + theta_step
    if p_start is None:
        p_start = 50.0

    if len(env_ids) == 0:
        stage = getattr(env, "_v5curr_stage", 0)
        return torch.tensor(float(stage), device=env.device)

    if not hasattr(env, "_v5curr_stage"):
        env._v5curr_stage = 0
        env._v5curr_theta_lo = theta_start_lo
        env._v5curr_theta_hi = theta_start_hi
        env._v5curr_p_lo = p_start
        env._v5curr_p_hi = p_start
        env._v5curr_dataset_pct = 0.0
        env._v5curr_window_done = 0
        env._v5curr_window_success = 0
        env._v5curr_last_eval_step = 0
        _apply_theta(env, env._v5curr_theta_lo, env._v5curr_theta_hi)
        _apply_p(env, env._v5curr_p_lo, env._v5curr_p_hi)

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
                _apply_theta(env, env._v5curr_theta_lo, env._v5curr_theta_hi)
                _apply_p(env, env._v5curr_p_lo, env._v5curr_p_hi)

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

    if "log" not in env.extras:
        env.extras["log"] = {}
    env.extras["log"]["Curriculum/v5_theta_lo"]     = float(getattr(env, "_v5curr_theta_lo", theta_mid - theta_step))
    env.extras["log"]["Curriculum/v5_theta_hi"]     = float(getattr(env, "_v5curr_theta_hi", theta_mid + theta_step))
    env.extras["log"]["Curriculum/v5_p_lo"]         = float(getattr(env, "_v5curr_p_lo", p_mid))
    env.extras["log"]["Curriculum/v5_p_hi"]         = float(getattr(env, "_v5curr_p_hi", p_mid))
    env.extras["log"]["Curriculum/v5_dataset_pct"]  = float(getattr(env, "_v5curr_dataset_pct", 0.0))
    # Canonical arm-init mixing fraction read by the unified reset_arm_mixed event.
    env._curr_dataset_pct = float(getattr(env, "_v5curr_dataset_pct", 0.0))
    return torch.tensor(float(env._v5curr_stage), device=env.device)


# ---------------------------------------------------------------------------
# Shared curriculum→event seam — both v5 (step-function) and v6 (PD) drive the
# same θ_init / p_des reset ranges through these helpers (mutate the reset-event
# params in place). Single definition; the advance *rule* is what varies, not
# how a range is applied.
# ---------------------------------------------------------------------------

def _apply_theta(env: "ManagerBasedRLEnv", lo: float, hi: float) -> None:
    cfg = env.event_manager.get_term_cfg("reset_valve_angle")
    cfg.params["angle_min"] = lo
    cfg.params["angle_max"] = hi


def _apply_p(env: "ManagerBasedRLEnv", lo: float, hi: float) -> None:
    cfg = env.event_manager.get_term_cfg("reset_p_des")
    cfg.params["p_min"] = lo
    cfg.params["p_max"] = hi


# ---------------------------------------------------------------------------
# v6 PD curriculum — decoupled θ/p expansion axes (ADR 0008)
# ---------------------------------------------------------------------------

def turn_pd_curriculum_v6(
    env: "ManagerBasedRLEnv",
    env_ids: Sequence[int],
    beta: float = 0.02,
    sr_target: float = 0.85,
    kp: float = 2.0,
    kd: float = 0.5,
    theta_scale: float = 1.0,
    p_scale: float = 4.625,
    mix_scale: float = 0.005,
    confirm_iters: int = 20,
    num_steps_per_env: int = 24,
    theta_min: float = 9.42,
    theta_max: float = 50.27,
    theta_start_hi: float | None = None,
    p_min: float = 15.0,
    p_max: float = 200.0,
    p_mid: float = 107.0,
) -> torch.Tensor:
    """v6 PD curriculum: decoupled θ/p expansion axes + dataset arm-init mixing.

    Replaces v5's step-function triggers with a continuous PD controller per axis.
    Stages are sequential (no cross-axis interaction):

      Stage 0 — θ expansion only. p fixed at p_mid=107 PSI.
        PD controller drives θ_lo↓ and θ_hi↑ toward [θ_min, θ_max].
        Advance when θ fully open AND EMA_SR ≥ sr_target for confirm_iters iters.

      Stage 1 — p expansion only. θ at full range.
        PD controller drives p_lo↓ and p_hi↑ around p_mid.
        Advance when p fully open AND EMA_SR ≥ sr_target for confirm_iters iters.

      Stage 2 — dataset arm-init mixing. θ and p fully open.
        PD controller drives dataset_pct 0.0→1.0.
        Advance when dataset_pct ≥ 1.0 AND EMA_SR ≥ sr_target for confirm_iters iters.

      Stage 3 — terminal. All axes fully open, 100% dataset init. No PD active.

    PD equations (per iteration):
        EMA_SR(t)  = β × SR_batch(t) + (1−β) × EMA_SR(t−1)
        error(t)   = EMA_SR(t) − sr_target
        d_error(t) = error(t) − error(t−1)
        delta(t)   = Kp × error(t) + Kd × d_error(t)

    δ > 0 → expand axis (SR above target). δ < 0 → contract (SR below target).

    State stored on env (prefix _v6curr_):
        stage, theta_lo, theta_hi, p_lo, p_hi, dataset_pct,
        ema_sr, prev_error, delta, confirm_count,
        window_done, window_success, last_eval_step.

    See docs/adr/0008-v6-pd-curriculum-decoupled-axes.md for full rationale.

    Args:
        beta:              EMA decay. Default 0.02 (~50-ep smoothing at 8192 envs).
        sr_target:         SR setpoint for PD and stage-advance check. Default 0.85.
        kp:                Proportional gain. Default 2.0.
        kd:                Derivative gain. Default 0.5.
        theta_scale:       rad/iter per unit delta for θ boundary movement. Default 1.0.
        p_scale:           PSI/iter per unit delta for p boundary movement. Default 4.625.
        mix_scale:         dataset_pct/iter per unit delta for mixing. Default 0.005.
        confirm_iters:     Consecutive iters at full+SR to advance stage. Default 20.
        num_steps_per_env: RSL-RL steps per env per iter (for iter cadence). Default 24.
        theta_min:         Minimum θ_init (rad). Default 9.42.
        theta_max:         Maximum θ_init (rad). Default 50.27.
        theta_start_hi:    Starting θ_hi. Default theta_min + 2.04 (5% of span).
        p_min:             Minimum p_des (PSI). Default 15.0.
        p_max:             Maximum p_des (PSI). Default 200.0.
        p_mid:             p_des fixed during Stage 0; symmetric centre in Stage 1. Default 107.0.

    Returns:
        Scalar tensor — current stage (0/1/2/3), for TensorBoard logging.
    """
    _THETA_STEP_DEFAULT: float = 2.04  # 5% of span (50.27 − 9.42 = 40.85 rad)

    if theta_start_hi is None:
        theta_start_hi = theta_min + _THETA_STEP_DEFAULT

    if len(env_ids) == 0:
        stage = getattr(env, "_v6curr_stage", 0)
        _v6_log(env, theta_min, p_mid)
        return torch.tensor(float(stage), device=env.device)

    # Initialise tracking state on first call
    if not hasattr(env, "_v6curr_stage"):
        env._v6curr_stage = 0
        env._v6curr_theta_lo = theta_min
        env._v6curr_theta_hi = float(theta_start_hi)
        env._v6curr_p_lo = p_mid
        env._v6curr_p_hi = p_mid
        env._v6curr_dataset_pct = 0.0
        env._v6curr_ema_sr = sr_target   # start neutral → delta = 0
        env._v6curr_prev_error = 0.0
        env._v6curr_delta = 0.0
        env._v6curr_confirm_count = 0
        env._v6curr_window_done = 0
        env._v6curr_window_success = 0
        env._v6curr_last_eval_step = 0
        _apply_theta(env, env._v6curr_theta_lo, env._v6curr_theta_hi)
        _apply_p(env, env._v6curr_p_lo, env._v6curr_p_hi)

    # Accumulate episode outcomes
    reset_terminated = getattr(env, "reset_terminated", None)
    n_success = int(reset_terminated[env_ids].sum().item()) if reset_terminated is not None else 0
    env._v6curr_window_done += len(env_ids)
    env._v6curr_window_success += n_success

    # Fire once per training iteration
    steps_since_eval = env.common_step_counter - env._v6curr_last_eval_step

    if steps_since_eval >= num_steps_per_env and env._v6curr_stage < 3:
        sr_batch = env._v6curr_window_success / max(env._v6curr_window_done, 1)

        # EMA update
        env._v6curr_ema_sr = beta * sr_batch + (1.0 - beta) * env._v6curr_ema_sr

        # PD compute
        error = env._v6curr_ema_sr - sr_target
        d_error = error - env._v6curr_prev_error
        delta = kp * error + kd * d_error
        env._v6curr_prev_error = error
        env._v6curr_delta = delta

        if env._v6curr_stage == 0:
            # Expand θ boundaries; both clamped to valid range
            env._v6curr_theta_hi = float(min(
                theta_max,
                max(env._v6curr_theta_lo + 0.1, env._v6curr_theta_hi + delta * theta_scale),
            ))
            env._v6curr_theta_lo = float(max(
                theta_min,
                min(env._v6curr_theta_hi - 0.1, env._v6curr_theta_lo - delta * theta_scale),
            ))
            _apply_theta(env, env._v6curr_theta_lo, env._v6curr_theta_hi)

            theta_full = (
                env._v6curr_theta_lo <= theta_min + 0.01
                and env._v6curr_theta_hi >= theta_max - 0.01
            )
            if theta_full and env._v6curr_ema_sr >= sr_target:
                env._v6curr_confirm_count += 1
            else:
                env._v6curr_confirm_count = 0

            print(
                f"[V6Curr] S0 | EMA_SR={env._v6curr_ema_sr:.3f} δ={delta:.3f} "
                f"θ=[{env._v6curr_theta_lo:.2f},{env._v6curr_theta_hi:.2f}] "
                f"confirm={env._v6curr_confirm_count}/{confirm_iters} "
                f"(step {env.common_step_counter})"
            )

            if env._v6curr_confirm_count >= confirm_iters:
                env._v6curr_stage = 1
                env._v6curr_confirm_count = 0
                env._v6curr_ema_sr = sr_target   # reset to neutral for new axis
                env._v6curr_prev_error = 0.0
                print(
                    f"[V6Curr] Stage 0→1 | θ full open | "
                    f"(step {env.common_step_counter})"
                )

        elif env._v6curr_stage == 1:
            # Expand p symmetrically around p_mid
            env._v6curr_p_hi = float(min(p_max, max(p_mid, env._v6curr_p_hi + delta * p_scale)))
            env._v6curr_p_lo = float(max(p_min, min(p_mid, env._v6curr_p_lo - delta * p_scale)))
            _apply_p(env, env._v6curr_p_lo, env._v6curr_p_hi)

            p_full = (
                env._v6curr_p_lo <= p_min + 0.1
                and env._v6curr_p_hi >= p_max - 0.1
            )
            if p_full and env._v6curr_ema_sr >= sr_target:
                env._v6curr_confirm_count += 1
            else:
                env._v6curr_confirm_count = 0

            print(
                f"[V6Curr] S1 | EMA_SR={env._v6curr_ema_sr:.3f} δ={delta:.3f} "
                f"p=[{env._v6curr_p_lo:.1f},{env._v6curr_p_hi:.1f}] "
                f"confirm={env._v6curr_confirm_count}/{confirm_iters} "
                f"(step {env.common_step_counter})"
            )

            if env._v6curr_confirm_count >= confirm_iters:
                env._v6curr_stage = 2
                env._v6curr_confirm_count = 0
                env._v6curr_ema_sr = sr_target
                env._v6curr_prev_error = 0.0
                print(
                    f"[V6Curr] Stage 1→2 | p full open | "
                    f"(step {env.common_step_counter})"
                )

        elif env._v6curr_stage == 2:
            env._v6curr_dataset_pct = float(
                max(0.0, min(1.0, env._v6curr_dataset_pct + delta * mix_scale))
            )

            mix_full = env._v6curr_dataset_pct >= 1.0 - 1e-6
            if mix_full and env._v6curr_ema_sr >= sr_target:
                env._v6curr_confirm_count += 1
            else:
                env._v6curr_confirm_count = 0

            print(
                f"[V6Curr] S2 | EMA_SR={env._v6curr_ema_sr:.3f} δ={delta:.3f} "
                f"mix={env._v6curr_dataset_pct:.1%} "
                f"confirm={env._v6curr_confirm_count}/{confirm_iters} "
                f"(step {env.common_step_counter})"
            )

            if env._v6curr_confirm_count >= confirm_iters:
                env._v6curr_stage = 3
                env._v6curr_confirm_count = 0
                print(
                    f"[V6Curr] Stage 2→3 | 100% dataset | "
                    f"(step {env.common_step_counter})"
                )

        # Reset iter accumulators
        env._v6curr_window_done = 0
        env._v6curr_window_success = 0
        env._v6curr_last_eval_step = env.common_step_counter

    _v6_log(env, theta_min, p_mid)
    return torch.tensor(float(env._v6curr_stage), device=env.device)


def _v6_log(env: "ManagerBasedRLEnv", theta_min: float, p_mid: float) -> None:
    """Write v6 curriculum scalars to env.extras['log'] for TensorBoard."""
    if "log" not in env.extras:
        env.extras["log"] = {}
    env.extras["log"]["Curriculum/v5_theta_lo"]    = float(getattr(env, "_v6curr_theta_lo",    theta_min))
    env.extras["log"]["Curriculum/v5_theta_hi"]    = float(getattr(env, "_v6curr_theta_hi",    theta_min))
    env.extras["log"]["Curriculum/v5_p_lo"]        = float(getattr(env, "_v6curr_p_lo",        p_mid))
    env.extras["log"]["Curriculum/v5_p_hi"]        = float(getattr(env, "_v6curr_p_hi",        p_mid))
    env.extras["log"]["Curriculum/v5_dataset_pct"] = float(getattr(env, "_v6curr_dataset_pct", 0.0))
    # v6-only PD diagnostics
    env.extras["log"]["Curriculum/ema_sr"]          = float(getattr(env, "_v6curr_ema_sr",      0.0))
    env.extras["log"]["Curriculum/delta"]           = float(getattr(env, "_v6curr_delta",       0.0))
    # Canonical arm-init mixing fraction read by the unified reset_arm_mixed event.
    env._curr_dataset_pct = float(getattr(env, "_v6curr_dataset_pct", 0.0))


# ---------------------------------------------------------------------------
# v7 PD curriculum — independent per-axis PD controllers (ADR 0012)
#
# Differences from v6:
#   - Two fully independent EMA+PD controllers, one per axis (θ and p).
#     v6 reuses a single EMA signal across stages; v7 keeps each EMA alive
#     and independent throughout training for diagnostics.
#   - No dataset-mixing stage.  v7 (Option-X) uses pre-grasp init throughout;
#     the "arm init" dimension is fixed at pre-grip and not advanced by the
#     curriculum.  This gives a cleaner two-axis ablation surface.
#   - Per-axis Kp/Kd/scale exposed separately so each axis can be tuned
#     without touching the other.
#
# Stage 0: θ expansion  (p fixed at p_mid).  θ-PD active; p-PD idle.
# Stage 1: p expansion  (θ fully open).      p-PD active; θ-PD idle.
# Stage 2: terminal — both axes fully open, pre-grasp init throughout.
# ---------------------------------------------------------------------------

def turn_pd_curriculum_v7(
    env: "ManagerBasedRLEnv",
    env_ids: Sequence[int],
    # --- θ-axis PD knobs ---
    kp_theta: float = 2.0,
    kd_theta: float = 0.5,
    beta_theta: float = 0.02,
    theta_scale: float = 1.0,
    # --- p-axis PD knobs ---
    kp_p: float = 2.0,
    kd_p: float = 0.5,
    beta_p: float = 0.02,
    p_scale: float = 4.625,
    # --- shared knobs ---
    sr_target: float = 0.85,
    confirm_iters: int = 20,
    num_steps_per_env: int = 24,
    # --- envelope ---
    theta_min: float = 9.42,
    theta_max: float = 50.27,
    theta_start_hi: float | None = None,
    p_min: float = 15.0,
    p_max: float = 200.0,
    p_mid: float = 107.0,
) -> torch.Tensor:
    """v7 independent-axis PD curriculum (no dataset mixing).

    Each axis has its own EMA success-rate tracker and PD controller.
    Stages are sequential; the active axis is advanced by its own controller
    while the idle axis's EMA is kept updated (for TensorBoard) but its
    boundaries are held fixed.

    Stage 0 — θ expansion only.
        p fixed at p_mid.  θ-PD drives θ_lo↓ and θ_hi↑ from a narrow
        starting window.  Advance when θ fully open AND θ-EMA_SR ≥ sr_target
        for confirm_iters consecutive iterations.

    Stage 1 — p expansion only.
        θ fully open (boundaries frozen).  p-PD drives p_lo↓ and p_hi↑
        around p_mid.  Advance when p fully open AND p-EMA_SR ≥ sr_target
        for confirm_iters consecutive iterations.

    Stage 2 — terminal.  Both axes fully open.  No arm-init mixing (v7 keeps
        pre-grasp throughout).

    PD equations (per iteration, per active axis):
        EMA_SR(t)  = β × SR_batch(t) + (1−β) × EMA_SR(t−1)
        error(t)   = EMA_SR(t) − sr_target
        d_error(t) = error(t) − error(t−1)
        delta(t)   = Kp × error(t) + Kd × d_error(t)

    Both EMA trackers are updated every iteration (regardless of which stage is
    active) so TensorBoard shows each axis's SR independently.

    Logged TensorBoard keys (env.extras['log']):
        Curriculum/stage             — 0 / 1 / 2
        Curriculum/theta_lo          — current θ_lo  (rad)
        Curriculum/theta_hi          — current θ_hi  (rad)
        Curriculum/p_lo              — current p_lo  (PSI)
        Curriculum/p_hi              — current p_hi  (PSI)
        Curriculum/ema_sr_theta      — θ-axis EMA success rate
        Curriculum/ema_sr_p          — p-axis EMA success rate
        Curriculum/delta_theta       — θ-axis PD output δ (bare name: overlays v5/v6 Curriculum/delta)
        Curriculum/delta_p           — p-axis PD output δ (bare name: overlays v5/v6 Curriculum/delta)
        Curriculum/confirm           — confirm-iter counter for active stage

    Note: ``env._curr_dataset_pct`` is set to 0.0 each call because v7 does
    not advance arm-init mixing; ``reset_arm_mixed`` (if present) will always
    use pre-grip for v7.

    Args:
        kp_theta:          Proportional gain for θ-axis PD. Default 2.0.
        kd_theta:          Derivative gain for θ-axis PD. Default 0.5.
        beta_theta:        EMA decay for θ-axis. Default 0.02 (~50-iter lag).
        theta_scale:       rad/iter per unit delta for θ boundary. Default 1.0.
        kp_p:              Proportional gain for p-axis PD. Default 2.0.
        kd_p:              Derivative gain for p-axis PD. Default 0.5.
        beta_p:            EMA decay for p-axis. Default 0.02.
        p_scale:           PSI/iter per unit delta for p boundary. Default 4.625
                           (same time-constant as θ at identical Kp/Kd).
        sr_target:         SR setpoint for both axes. Default 0.85.
        confirm_iters:     Consecutive iters at full+SR to advance. Default 20.
        num_steps_per_env: RSL-RL steps per env per iter. Default 24.
        theta_min:         Minimum θ_init (rad). Default 9.42 (1.5 rev).
        theta_max:         Maximum θ_init (rad). Default 50.27 (8 rev).
        theta_start_hi:    Starting θ_hi. Default theta_min + 2.04 (5% of span).
        p_min:             Minimum p_des (PSI). Default 15.0.
        p_max:             Maximum p_des (PSI). Default 200.0.
        p_mid:             Fixed p during Stage 0; symmetric centre in Stage 1.
                           Default 107.0.

    Returns:
        Scalar tensor — current stage (0/1/2), for TensorBoard logging.
    """
    _THETA_STEP_DEFAULT: float = 2.04  # 5% of span (50.27 − 9.42 = 40.85 rad)

    if theta_start_hi is None:
        theta_start_hi = theta_min + _THETA_STEP_DEFAULT

    if len(env_ids) == 0:
        _v7_log(env, theta_min, p_mid)
        return torch.tensor(float(getattr(env, "_v7curr_stage", 0)), device=env.device)

    # Initialise per-axis tracking state on first call
    if not hasattr(env, "_v7curr_stage"):
        env._v7curr_stage = 0
        # θ-axis state
        env._v7curr_theta_lo = theta_min
        env._v7curr_theta_hi = float(theta_start_hi)
        env._v7curr_ema_theta = sr_target   # start neutral → delta = 0
        env._v7curr_prev_err_theta = 0.0
        env._v7curr_delta_theta = 0.0
        # p-axis state
        env._v7curr_p_lo = p_mid
        env._v7curr_p_hi = p_mid
        env._v7curr_ema_p = sr_target
        env._v7curr_prev_err_p = 0.0
        env._v7curr_delta_p = 0.0
        # shared
        env._v7curr_confirm_count = 0
        env._v7curr_window_done = 0
        env._v7curr_window_success = 0
        env._v7curr_last_eval_step = 0
        # apply initial ranges
        _apply_theta(env, env._v7curr_theta_lo, env._v7curr_theta_hi)
        _apply_p(env, env._v7curr_p_lo, env._v7curr_p_hi)

    # Accumulate episode outcomes
    reset_terminated = getattr(env, "reset_terminated", None)
    n_success = int(reset_terminated[env_ids].sum().item()) if reset_terminated is not None else 0
    env._v7curr_window_done += len(env_ids)
    env._v7curr_window_success += n_success

    # Fire once per training iteration
    steps_since_eval = env.common_step_counter - env._v7curr_last_eval_step

    if steps_since_eval >= num_steps_per_env and env._v7curr_stage < 2:
        sr_batch = env._v7curr_window_success / max(env._v7curr_window_done, 1)

        # Update BOTH EMA trackers every iteration (independent signals)
        env._v7curr_ema_theta = beta_theta * sr_batch + (1.0 - beta_theta) * env._v7curr_ema_theta
        env._v7curr_ema_p = beta_p * sr_batch + (1.0 - beta_p) * env._v7curr_ema_p

        # --- θ-axis PD (active only in Stage 0) ---
        err_theta = env._v7curr_ema_theta - sr_target
        d_err_theta = err_theta - env._v7curr_prev_err_theta
        delta_theta = kp_theta * err_theta + kd_theta * d_err_theta
        env._v7curr_prev_err_theta = err_theta
        env._v7curr_delta_theta = delta_theta

        # --- p-axis PD (active only in Stage 1) ---
        err_p = env._v7curr_ema_p - sr_target
        d_err_p = err_p - env._v7curr_prev_err_p
        delta_p = kp_p * err_p + kd_p * d_err_p
        env._v7curr_prev_err_p = err_p
        env._v7curr_delta_p = delta_p

        if env._v7curr_stage == 0:
            # Apply θ-axis PD
            env._v7curr_theta_hi = float(min(
                theta_max,
                max(env._v7curr_theta_lo + 0.1, env._v7curr_theta_hi + delta_theta * theta_scale),
            ))
            env._v7curr_theta_lo = float(max(
                theta_min,
                min(env._v7curr_theta_hi - 0.1, env._v7curr_theta_lo - delta_theta * theta_scale),
            ))
            _apply_theta(env, env._v7curr_theta_lo, env._v7curr_theta_hi)

            theta_full = (
                env._v7curr_theta_lo <= theta_min + 0.01
                and env._v7curr_theta_hi >= theta_max - 0.01
            )
            if theta_full and env._v7curr_ema_theta >= sr_target:
                env._v7curr_confirm_count += 1
            else:
                env._v7curr_confirm_count = 0

            print(
                f"[V7Curr] S0/θ | ema_θ={env._v7curr_ema_theta:.3f} "
                f"δ_θ={delta_theta:.3f} "
                f"θ=[{env._v7curr_theta_lo:.2f},{env._v7curr_theta_hi:.2f}] "
                f"ema_p={env._v7curr_ema_p:.3f} "
                f"confirm={env._v7curr_confirm_count}/{confirm_iters} "
                f"(step {env.common_step_counter})"
            )

            if env._v7curr_confirm_count >= confirm_iters:
                env._v7curr_stage = 1
                env._v7curr_confirm_count = 0
                # Reset p-axis EMA to neutral for fresh p-axis start
                env._v7curr_ema_p = sr_target
                env._v7curr_prev_err_p = 0.0
                print(
                    f"[V7Curr] Stage 0→1 | θ fully open | "
                    f"(step {env.common_step_counter})"
                )

        elif env._v7curr_stage == 1:
            # Apply p-axis PD
            env._v7curr_p_hi = float(min(p_max, max(p_mid, env._v7curr_p_hi + delta_p * p_scale)))
            env._v7curr_p_lo = float(max(p_min, min(p_mid, env._v7curr_p_lo - delta_p * p_scale)))
            _apply_p(env, env._v7curr_p_lo, env._v7curr_p_hi)

            p_full = (
                env._v7curr_p_lo <= p_min + 0.1
                and env._v7curr_p_hi >= p_max - 0.1
            )
            if p_full and env._v7curr_ema_p >= sr_target:
                env._v7curr_confirm_count += 1
            else:
                env._v7curr_confirm_count = 0

            print(
                f"[V7Curr] S1/p | ema_p={env._v7curr_ema_p:.3f} "
                f"δ_p={delta_p:.3f} "
                f"p=[{env._v7curr_p_lo:.1f},{env._v7curr_p_hi:.1f}] "
                f"ema_θ={env._v7curr_ema_theta:.3f} "
                f"confirm={env._v7curr_confirm_count}/{confirm_iters} "
                f"(step {env.common_step_counter})"
            )

            if env._v7curr_confirm_count >= confirm_iters:
                env._v7curr_stage = 2
                env._v7curr_confirm_count = 0
                print(
                    f"[V7Curr] Stage 1→2 | p fully open | terminal "
                    f"(step {env.common_step_counter})"
                )

        # Reset iter accumulators
        env._v7curr_window_done = 0
        env._v7curr_window_success = 0
        env._v7curr_last_eval_step = env.common_step_counter

    # v7 never advances arm-init mixing; canonical pct stays 0.
    env._curr_dataset_pct = 0.0
    _v7_log(env, theta_min, p_mid)
    return torch.tensor(float(env._v7curr_stage), device=env.device)


def _v7_log(env: "ManagerBasedRLEnv", theta_min: float, p_mid: float) -> None:
    """Write v7 curriculum scalars to env.extras['log'] for TensorBoard."""
    if "log" not in env.extras:
        env.extras["log"] = {}
    env.extras["log"]["Curriculum/stage"]           = float(getattr(env, "_v7curr_stage",       0))
    env.extras["log"]["Curriculum/theta_lo"]        = float(getattr(env, "_v7curr_theta_lo",    theta_min))
    env.extras["log"]["Curriculum/theta_hi"]        = float(getattr(env, "_v7curr_theta_hi",    theta_min))
    env.extras["log"]["Curriculum/p_lo"]            = float(getattr(env, "_v7curr_p_lo",        p_mid))
    env.extras["log"]["Curriculum/p_hi"]            = float(getattr(env, "_v7curr_p_hi",        p_mid))
    env.extras["log"]["Curriculum/ema_sr_theta"]    = float(getattr(env, "_v7curr_ema_theta",   0.0))
    env.extras["log"]["Curriculum/ema_sr_p"]        = float(getattr(env, "_v7curr_ema_p",       0.0))
    env.extras["log"]["Curriculum/delta_theta"]     = float(getattr(env, "_v7curr_delta_theta", 0.0))
    env.extras["log"]["Curriculum/delta_p"]         = float(getattr(env, "_v7curr_delta_p",     0.0))
    env.extras["log"]["Curriculum/confirm"]         = float(getattr(env, "_v7curr_confirm_count", 0))
