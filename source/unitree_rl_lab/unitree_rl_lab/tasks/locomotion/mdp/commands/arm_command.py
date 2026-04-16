from __future__ import annotations

import torch
from dataclasses import MISSING
from typing import TYPE_CHECKING, Sequence

from isaaclab.managers import CommandTermCfg, CommandTerm
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class UniformArmPoseCommand(CommandTerm):
    """Samples target shoulder_pitch angle independently for left and right arm.

    Command tensor: [left_shoulder_pitch_target, right_shoulder_pitch_target]
    Range sampled uniformly from cfg.ranges each episode.
    """

    cfg: UniformArmPoseCommandCfg

    def __init__(self, cfg: UniformArmPoseCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._command = torch.zeros(env.num_envs, 2, device=self.device)

    def __str__(self) -> str:
        return (
            f"UniformArmPoseCommand(\n"
            f"  left  range: {self.cfg.ranges.left_shoulder_pitch}\n"
            f"  right range: {self.cfg.ranges.right_shoulder_pitch}\n"
            f")"
        )

    @property
    def command(self) -> torch.Tensor:
        return self._command

    def _update_command(self):
        pass  # command held fixed until resampled

    def _resample_command(self, env_ids: Sequence[int]):
        lo_l, hi_l = self.cfg.ranges.left_shoulder_pitch
        lo_r, hi_r = self.cfg.ranges.right_shoulder_pitch
        self._command[env_ids, 0].uniform_(lo_l, hi_l)
        self._command[env_ids, 1].uniform_(lo_r, hi_r)

    def _update_metrics(self):
        pass

    def _set_debug_vis_impl(self, debug_vis: bool):
        pass

    def _debug_vis_callback(self, event):
        pass


@configclass
class UniformArmPoseCommandCfg(CommandTermCfg):
    """Configuration for UniformArmPoseCommand."""

    class_type: type = UniformArmPoseCommand

    @configclass
    class Ranges:
        left_shoulder_pitch: tuple[float, float] = MISSING
        right_shoulder_pitch: tuple[float, float] = MISSING

    ranges: Ranges = MISSING
