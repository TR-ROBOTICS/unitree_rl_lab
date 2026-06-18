"""Module #10 — Turn-preset parity gate (ADR 0009 item 5).

After the v0→v6b subclass ladder was collapsed into one parameterized
`TurnEnvCfg` + the flat `TURN_PRESETS` catalog (valve/presets.py), this test
asserts every preset *builds* and that the assembled managers match the spec —
catching the silent failure mode of a builder refactor (a wrong weight, a
dropped term, a mismatched range, a reordered event) that `py_compile` cannot.

Requires a headless IsaacSim launch (the cfg modules import isaaclab → pxr).
Building a cfg does NOT spawn physics, so the parity checks are fast once the app
is up. The opt-in `gym.make` smoke (RUN_GYM_MAKE=1) does spawn one env per task.

Run:
    conda run -n env_isaaclab python -m pytest \
        source/unitree_rl_lab/tests/manipulation/test_10_turn_presets.py -v

    # include the end-to-end env-creation smoke (slower, needs GPU physx):
    RUN_GYM_MAKE=1 conda run -n env_isaaclab python -m pytest \
        source/unitree_rl_lab/tests/manipulation/test_10_turn_presets.py -v
"""

from __future__ import annotations

import os

import pytest

# ---------------------------------------------------------------------------
# Launch IsaacSim headless before importing isaaclab-dependent cfg modules.
# Skip the whole module if the app is unavailable (e.g. CI box without GPU/Kit).
# ---------------------------------------------------------------------------
try:
    from isaaclab.app import AppLauncher

    _APP = AppLauncher(headless=True).app
except Exception as exc:  # pragma: no cover - environment dependent
    pytest.skip(f"IsaacSim app unavailable: {exc}", allow_module_level=True)

import unitree_rl_lab.tasks.manipulation.mdp as mdp  # noqa: E402
from unitree_rl_lab.tasks.manipulation.robots.g1.dof29.valve import presets  # noqa: E402

NAMES = list(presets.TURN_PRESETS)

_ARM_FUNC = {
    "dataset": "reset_arm_from_dataset",
    "staged": "reset_arm_staged",
    "mixed": "reset_arm_mixed",
}
_CURR_FUNC = {
    "auto": "turn_auto_curriculum_stage",
    "auto_easy": "turn_auto_curriculum_stage_easy",
    "smooth_v5": "turn_smooth_curriculum_v5",
    "pd_v6": "turn_pd_curriculum_v6",
    "pd_v7": "turn_pd_curriculum_v7",
}


def _curr_func_name(curriculum_cfg) -> str | None:
    """Return the func name of the single CurrTerm on a curriculum configclass."""
    if curriculum_cfg is None:
        return None
    for term in vars(curriculum_cfg).values():
        if hasattr(term, "func"):
            return term.func.__name__
    return None


# ---------------------------------------------------------------------------
# Build smoke — every preset assembles without error
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", NAMES)
def test_env_builds(name):
    cfg = presets.build_env_cfg(name)
    assert cfg.scene.num_envs == 4096
    assert cfg.observations is not None
    assert cfg.actions is not None
    assert cfg.rewards is not None
    assert cfg.terminations is not None
    assert cfg.events is not None


@pytest.mark.parametrize("name", NAMES)
def test_play_builds(name):
    cfg = presets.build_play_cfg(name)
    assert cfg.scene.num_envs == 1


# ---------------------------------------------------------------------------
# g(θ) configclass fields — contract read by scripts/rsl_rl/probe_reset_direction.py
# ---------------------------------------------------------------------------

def test_pressure_fields():
    cfg = presets.build_env_cfg("v0")
    assert cfg.pressure_a == mdp.pressure.A
    assert cfg.pressure_b == mdp.pressure.B
    assert cfg.theta_min == mdp.pressure.THETA_MIN
    assert cfg.theta_max == mdp.pressure.THETA_MAX
    assert cfg.p_des_range == (107.0, 107.0)


# ---------------------------------------------------------------------------
# Event parity — θ/p ranges, DR-disable, arm-init func match the spec
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", NAMES)
def test_event_parity(name):
    spec = presets.TURN_PRESETS[name]
    ev = presets.build_env_cfg(name).events

    assert ev.reset_valve_angle.params["angle_min"] == spec.theta_init[0]
    assert ev.reset_valve_angle.params["angle_max"] == spec.theta_init[1]
    assert ev.reset_p_des.params["p_min"] == spec.p_des[0]
    assert ev.reset_p_des.params["p_max"] == spec.p_des[1]
    # Valve-pos DR disabled for bootstrapping (was ValveTurnEnvCfg.__post_init__)
    assert ev.reset_valve_pos.params["half_range_xyz"] == (0.0, 0.0, 0.0)

    if spec.arm_init == "pregrip":
        assert ev.reset_arm is None
    else:
        assert ev.reset_arm is not None
        assert ev.reset_arm.func.__name__ == _ARM_FUNC[spec.arm_init]


# ---------------------------------------------------------------------------
# Reward parity — optional terms present iff the spec flag is set
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", NAMES)
def test_reward_parity(name):
    spec = presets.TURN_PRESETS[name]
    r = presets.build_env_cfg(name).rewards

    # Always-on pressure terms
    assert r.pressure_error is not None and r.pressure_error.weight == 0.2
    assert r.pressure_progress is not None and r.pressure_progress.weight == 30.0

    assert (r.action_rate is not None) == spec.smoothness
    if spec.smoothness:
        assert r.action_rate.weight == -0.0001

    assert (r.bilateral_contact is not None) == spec.contact_rewards
    assert (r.contact_force_jerk is not None) == spec.contact_rewards

    assert (r.bimanual_progress is not None) == spec.bimanual_rewards
    assert (r.single_hand_turning_penalty is not None) == spec.bimanual_rewards
    if spec.bimanual_rewards:
        assert r.bimanual_progress.weight == 30.0
        assert r.single_hand_turning_penalty.weight == -15.0


# ---------------------------------------------------------------------------
# Curriculum / action / obs parity
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", NAMES)
def test_curriculum_parity(name):
    spec = presets.TURN_PRESETS[name]
    cfg = presets.build_env_cfg(name)

    assert (cfg.curriculum is None) == (spec.curriculum is None)
    if spec.curriculum is not None:
        assert _curr_func_name(cfg.curriculum) == _CURR_FUNC[spec.curriculum]


@pytest.mark.parametrize("name", NAMES)
def test_action_obs_space(name):
    spec = presets.TURN_PRESETS[name]
    cfg = presets.build_env_cfg(name)

    # 38-DoF hand action term present iff hands; finger obs term present iff hands
    assert hasattr(cfg.actions, "hands") == spec.hands
    assert hasattr(cfg.observations.policy, "finger_joint_pos_rel") == spec.hands


# ---------------------------------------------------------------------------
# Play parity — terminal distribution, curriculum bypass, num_envs
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", NAMES)
def test_play_distribution(name):
    spec = presets.TURN_PRESETS[name]
    play = presets.build_play_cfg(name)

    assert play.scene.num_envs == 1

    if spec.play.events == "terminal":
        assert play.events.reset_valve_angle.params["angle_min"] == mdp.pressure.THETA_MIN
        assert play.events.reset_valve_angle.params["angle_max"] == mdp.pressure.THETA_MAX
        assert play.events.reset_p_des.params["p_min"] == mdp.pressure.P_MIN
        assert play.events.reset_p_des.params["p_max"] == mdp.pressure.P_MAX
        # v7 keeps pre-grasp arm init (no dataset mixing); all other terminal presets use dataset.
        if spec.pregrasp_init:
            assert play.events.reset_arm is None  # pregrip: no reset_arm event
            assert hasattr(play.events, "reset_arm_pregrasp")  # explicit pre-grasp event present
        else:
            assert play.events.reset_arm.func.__name__ == "reset_arm_from_dataset"

    if spec.play.drop_curriculum:
        assert play.curriculum is None


def test_play_p_des_override():
    """apply_p_des presets honor VALVE_P_DES; others ignore it."""
    os.environ["VALVE_P_DES"] = "50"
    try:
        for name in NAMES:
            spec = presets.TURN_PRESETS[name]
            ev = presets.build_play_cfg(name).events
            if spec.play.apply_p_des:
                assert ev.reset_p_des.params["p_min"] == 50.0, name
                assert ev.reset_p_des.params["p_max"] == 50.0, name
            else:
                assert ev.reset_p_des.params["p_min"] != 50.0 or spec.p_des[0] == 50.0, name
    finally:
        del os.environ["VALVE_P_DES"]


# ---------------------------------------------------------------------------
# Opt-in end-to-end env creation (spawns physics) — guarded by RUN_GYM_MAKE=1
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", NAMES)
def test_pregrasp_init_parity(name):
    """Presets with pregrasp_init=True must have reset_arm_pregrasp event; others must not."""
    spec = presets.TURN_PRESETS[name]
    ev = presets.build_env_cfg(name).events
    if spec.pregrasp_init:
        assert hasattr(ev, "reset_arm_pregrasp"), (
            f"{name}: pregrasp_init=True but reset_arm_pregrasp EventTerm absent"
        )
        assert ev.reset_arm_pregrasp.func.__name__ == "reset_arm_pregrasp"
        assert ev.reset_arm_pregrasp.params["enabled"] is True
    else:
        assert not hasattr(ev, "reset_arm_pregrasp"), (
            f"{name}: pregrasp_init=False but reset_arm_pregrasp EventTerm present"
        )


@pytest.mark.skipif(
    os.environ.get("RUN_GYM_MAKE") != "1",
    reason="set RUN_GYM_MAKE=1 to spawn envs (slow; needs GPU physx)",
)
@pytest.mark.parametrize("name", ["v5", "v6", "v6b", "v7"])
def test_gym_make_play(name):
    """Live tasks must construct a real env + reset (validates joint-name resolution)."""
    import gymnasium as gym
    import unitree_rl_lab.tasks  # noqa: F401  — triggers gym registration

    from isaaclab_tasks.utils import parse_env_cfg

    task_id = f"Unitree-G1-29dof-ValveTurn-{name}"
    env_cfg = parse_env_cfg(task_id, num_envs=1)
    env = gym.make(task_id, cfg=env_cfg)
    try:
        obs, _ = env.reset()
        assert obs["policy"].shape[0] == 1
    finally:
        env.close()
