"""Shared pytest configuration for unitree_rl_lab manipulation tests."""

import pathlib
import sys

# Ensure tasks/ importable as top-level package (mirrors list_envs.py behaviour)
_TASKS_PATH = str(
    pathlib.Path(__file__).parents[2]
    / "unitree_rl_lab"
    / "tasks"
)
if _TASKS_PATH not in sys.path:
    sys.path.insert(0, _TASKS_PATH)
