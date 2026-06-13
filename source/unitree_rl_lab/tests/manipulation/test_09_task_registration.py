"""Module #9 — Task registration test.

PRD ref: docs/prd/IsaacLab-task.md §Testing Decisions "Task registration (module #9)"

Verifies that `Unitree-G1-29dof-ValveTurn-v0` appears in the output of
`scripts/list_envs.py` (the same path used by `./unitree_rl_lab.sh -l`).

No IsaacSim launch required — list_envs.py imports gymnasium only.

Run:
    conda run -n env_isaaclab pytest source/unitree_rl_lab/tests/manipulation/test_09_task_registration.py -v
"""

from __future__ import annotations

import importlib
import pathlib
import pkgutil
import subprocess
import sys

import pytest

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = pathlib.Path(__file__).parents[4]  # …/unitree_rl_lab/
LIST_ENVS_SCRIPT = REPO_ROOT / "scripts" / "list_envs.py"

TARGET_TASK_ID = "Unitree-G1-29dof-ValveTurn-v0"


# ---------------------------------------------------------------------------
# Helper — run list_envs via same interpreter used by pytest
# ---------------------------------------------------------------------------
def _run_list_envs() -> tuple[str, str, int]:
    """Returns (stdout, stderr, returncode)."""
    result = subprocess.run(
        [sys.executable, str(LIST_ENVS_SCRIPT)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    return result.stdout, result.stderr, result.returncode


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestTaskRegistration:
    """Module #9: Unitree-G1-29dof-ValveTurn-v0 gym registration."""

    def test_list_envs_exits_cleanly(self):
        """list_envs.py must exit 0."""
        _, stderr, rc = _run_list_envs()
        assert rc == 0, f"list_envs.py exited {rc}.\nstderr:\n{stderr}"

    def test_task_id_present(self):
        """Target task ID must appear in list_envs output."""
        stdout, stderr, rc = _run_list_envs()
        assert TARGET_TASK_ID in stdout, (
            f"{TARGET_TASK_ID!r} not found in list_envs output.\n"
            f"stdout:\n{stdout}\nstderr:\n{stderr}"
        )

    def test_entry_point_correct(self):
        """Entry point must be isaaclab.envs:ManagerBasedRLEnv."""
        stdout, _, _ = _run_list_envs()
        assert "isaaclab.envs:ManagerBasedRLEnv" in stdout

    def test_config_entry_point_correct(self):
        """Config entry point must reference ValveTurnEnvCfg."""
        stdout, _, _ = _run_list_envs()
        assert "ValveTurnEnvCfg" in stdout

    def test_gym_register_importable(self):
        """Direct import of dof29 __init__ must register task in gymnasium."""
        # Insert tasks/ onto path the same way list_envs.py does
        tasks_path = str(REPO_ROOT / "source" / "unitree_rl_lab" / "unitree_rl_lab" / "tasks")
        if tasks_path not in sys.path:
            sys.path.insert(0, tasks_path)

        # Walk and import manipulation.robots subtree
        pkg = importlib.import_module("manipulation.robots")
        for _ in pkgutil.walk_packages(pkg.__path__, pkg.__name__ + "."):
            pass

        import gymnasium as gym
        ids = [spec.id for spec in gym.registry.values()]
        assert TARGET_TASK_ID in ids, (
            f"{TARGET_TASK_ID!r} not in gymnasium registry after import.\n"
            f"Registered Unitree tasks: {[i for i in ids if 'Unitree' in i]}"
        )

