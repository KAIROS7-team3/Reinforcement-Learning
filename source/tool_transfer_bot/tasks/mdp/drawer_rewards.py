"""Reward functions for drawer open/close tasks (Task 1 and Task 4).

Adapted from isaaclab_tasks cabinet mdp/rewards.py.
Key difference: drawer joint is **prismatic/linear** (Y-axis).
  - closed = 0 m, fully open = -0.2 m  → drawer_pos is negative when open.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import matrix_from_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def approach_ee_handle(env: ManagerBasedRLEnv, threshold: float = 0.2) -> torch.Tensor:
    """Inverse-square reward for EE approaching the drawer handle."""
    ee_tcp_pos = env.scene["ee_frame"].data.target_pos_w[..., 0, :]
    handle_pos = env.scene["drawer_frame"].data.target_pos_w[..., 0, :]
    distance = torch.norm(handle_pos - ee_tcp_pos, dim=-1, p=2)
    reward = 1.0 / (1.0 + distance**2)
    reward = torch.pow(reward, 2)
    return torch.where(distance <= threshold, 2 * reward, reward)


def align_ee_handle(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Reward for aligning EE orientation with the drawer handle."""
    ee_tcp_quat = env.scene["ee_frame"].data.target_quat_w[..., 0, :]
    handle_quat = env.scene["drawer_frame"].data.target_quat_w[..., 0, :]

    ee_mat = matrix_from_quat(ee_tcp_quat)
    handle_mat = matrix_from_quat(handle_quat)

    handle_x, handle_y = handle_mat[..., 0], handle_mat[..., 1]
    ee_x, ee_z = ee_mat[..., 0], ee_mat[..., 2]

    align_z = torch.bmm(ee_z.unsqueeze(1), -handle_x.unsqueeze(-1)).squeeze(-1).squeeze(-1)
    align_x = torch.bmm(ee_x.unsqueeze(1), -handle_y.unsqueeze(-1)).squeeze(-1).squeeze(-1)
    return 0.5 * (torch.sign(align_z) * align_z**2 + torch.sign(align_x) * align_x**2)


def align_grasp_around_handle(env: ManagerBasedRLEnv) -> torch.Tensor:
    """True when left finger is above handle and right finger is below (graspable pose)."""
    handle_pos = env.scene["drawer_frame"].data.target_pos_w[..., 0, :]
    fingertips_w = env.scene["ee_frame"].data.target_pos_w[..., 1:, :]  # (N, 2, 3)
    lfinger_pos = fingertips_w[..., 0, :]
    rfinger_pos = fingertips_w[..., 1, :]
    return (rfinger_pos[:, 2] < handle_pos[:, 2]) & (lfinger_pos[:, 2] > handle_pos[:, 2])


def approach_gripper_handle(env: ManagerBasedRLEnv, offset: float = 0.04) -> torch.Tensor:
    """Distance reward for fingers closing around the handle when in graspable orientation."""
    handle_pos = env.scene["drawer_frame"].data.target_pos_w[..., 0, :]
    fingertips_w = env.scene["ee_frame"].data.target_pos_w[..., 1:, :]
    lfinger_pos = fingertips_w[..., 0, :]
    rfinger_pos = fingertips_w[..., 1, :]

    lfinger_dist = torch.abs(lfinger_pos[:, 2] - handle_pos[:, 2])
    rfinger_dist = torch.abs(rfinger_pos[:, 2] - handle_pos[:, 2])
    is_graspable = align_grasp_around_handle(env)
    return is_graspable * ((offset - lfinger_dist) + (offset - rfinger_dist))


def grasp_handle(
    env: ManagerBasedRLEnv,
    threshold: float,
    open_joint_pos: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Reward for closing fingers when EE is close to the handle."""
    ee_tcp_pos = env.scene["ee_frame"].data.target_pos_w[..., 0, :]
    handle_pos = env.scene["drawer_frame"].data.target_pos_w[..., 0, :]
    gripper_joint_pos = env.scene[asset_cfg.name].data.joint_pos[:, asset_cfg.joint_ids]
    distance = torch.norm(handle_pos - ee_tcp_pos, dim=-1, p=2)
    is_close = distance <= threshold
    return is_close * torch.sum(open_joint_pos - gripper_joint_pos, dim=-1)


def open_drawer_bonus(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Proportional bonus for drawer displacement (positive when drawer is open).

    Drawer joint: lowerLimit=-0.2 (fully open), upperLimit=0 (closed).
    We reward |pos| so the agent is motivated to pull the drawer.
    """
    drawer_pos = env.scene[asset_cfg.name].data.joint_pos[:, asset_cfg.joint_ids[0]]
    is_graspable = align_grasp_around_handle(env).float()
    return (is_graspable + 1.0) * torch.abs(drawer_pos)


def multi_stage_open_drawer(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Staged bonus at 1 cm, 10 cm, and 18 cm open.

    Thresholds use |drawer_pos| since joint goes negative when opening.
    Max travel is 0.2 m → stages at 0.01, 0.10, 0.18 m.
    """
    drawer_pos = torch.abs(env.scene[asset_cfg.name].data.joint_pos[:, asset_cfg.joint_ids[0]])
    is_graspable = align_grasp_around_handle(env).float()

    open_easy = (drawer_pos > 0.01).float() * 0.5
    open_medium = (drawer_pos > 0.10).float() * is_graspable
    open_hard = (drawer_pos > 0.18).float() * is_graspable
    return open_easy + open_medium + open_hard
