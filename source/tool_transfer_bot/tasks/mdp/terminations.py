"""Custom termination functions for tool-transfer tasks."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def task_success(
    env: ManagerBasedRLEnv,
    threshold: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("toolbox", joint_names=["drawer_joint"]),
) -> torch.Tensor:
    """Episode succeeds when |drawer_joint_pos| >= threshold.

    Drawer joint goes negative when open (0 → -0.2 m), so we use abs.
    """
    drawer_pos = env.scene[asset_cfg.name].data.joint_pos[:, asset_cfg.joint_ids[0]]
    return torch.abs(drawer_pos) >= threshold
