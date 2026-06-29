"""RH-P12 gripper: rh_r1 drives rh_l1 / rh_l2 / rh_r2 (same joint-space angle in Isaac Lab)."""

from __future__ import annotations

import torch
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import SceneEntityCfg

MIMIC_GEARING = 1.0
MIMIC_JOINT_NAMES = ("rh_r2", "rh_l1", "rh_l2")
DRIVEN_GRIPPER_JOINT = "rh_r1"


def sync_gripper_mimic_targets(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    driven_joint: str = DRIVEN_GRIPPER_JOINT,
    mimic_gearing: float = MIMIC_GEARING,
) -> None:
    """Set mimic finger PD targets from rh_r1 command (not measured finger drift)."""
    del env_ids  # all envs share the same rh_r1 target pattern
    robot = env.scene[asset_cfg.name]
    if driven_joint not in robot.joint_names:
        return

    r1_idx = robot.joint_names.index(driven_joint)
    r1 = robot.data.joint_pos_target[:, r1_idx]
    mimic = mimic_gearing * r1

    for name in MIMIC_JOINT_NAMES:
        if name not in robot.joint_names:
            continue
        idx = robot.joint_names.index(name)
        robot.set_joint_position_target(mimic.unsqueeze(-1), joint_ids=[idx])
