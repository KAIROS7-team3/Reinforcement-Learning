"""Reward functions for drawer open/close tasks (Task 1 and Task 4).

Adapted from isaaclab_tasks cabinet mdp/rewards.py.
Key difference: drawer joint is **prismatic/linear** (Y-axis).
  - closed = 0 m, fully open = -0.2 m  → drawer_pos is negative when open.

Grasp rewards use the drawer **knob frame** (``drawer_frame`` / ``FurnitureKnob_01`` center;
coplanar Z + Y straddle + X depth). Not prim ``handle`` (toolbox top carry handle) and not
the drawer-front pull appearance baked into ``drawer/drawer`` mesh texture.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import matrix_from_quat, quat_apply_inverse

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _fingertips_in_handle_frame(env: ManagerBasedRLEnv) -> tuple[torch.Tensor, torch.Tensor]:
    """Left/right fingertip positions in knob frame (``drawer_frame`` origin)."""
    handle_pos = env.scene["drawer_frame"].data.target_pos_w[..., 0, :]
    handle_quat = env.scene["drawer_frame"].data.target_quat_w[..., 0, :]
    fingertips_w = env.scene["ee_frame"].data.target_pos_w[..., 1:, :]  # (N, 2, 3)

    rel_w = fingertips_w - handle_pos.unsqueeze(1)
    rel_h = quat_apply_inverse(handle_quat.unsqueeze(1).expand(-1, 2, -1), rel_w)
    return rel_h[..., 0, :], rel_h[..., 1, :]


def _joint_pos_scalar(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """One joint angle per env (rad). Works with resolved or teleop-unresolved SceneEntityCfg."""
    asset = env.scene[asset_cfg.name]
    if asset_cfg.joint_names is not None:
        name = asset_cfg.joint_names[0]
        if len(asset_cfg.joint_names) != 1:
            raise ValueError(f"expected one joint name, got {asset_cfg.joint_names}")
        idx = asset.joint_names.index(name)
        return asset.data.joint_pos[:, idx]
    pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    return pos.squeeze(-1)


def _gripper_joint_pos(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Scalar gripper joint angle per env (rad). Shape: (num_envs,)."""
    return _joint_pos_scalar(env, asset_cfg)


def _gripper_close_norm(
    gripper_joint_pos: torch.Tensor,
    open_joint_pos: float,
    close_joint_pos: float,
) -> torch.Tensor:
    """RH-P12: 0 rad = open, 60 deg = closed → normalized close amount in [0, 1]."""
    close_amount = gripper_joint_pos - open_joint_pos
    close_span = close_joint_pos - open_joint_pos
    close_amount = torch.clamp(close_amount, min=0.0, max=close_span)
    return close_amount / (close_span + 1e-6)


def _gripper_is_closed(
    env: ManagerBasedRLEnv,
    gripper_asset_cfg: SceneEntityCfg,
    close_threshold: float,
) -> torch.Tensor:
    """True when rh_r1 is past the pinch threshold (drawer pull requires closed gripper)."""
    return _gripper_joint_pos(env, gripper_asset_cfg) >= close_threshold


def _ee_handle_distance(env: ManagerBasedRLEnv) -> torch.Tensor:
    """EE TCP to drawer knob distance (m). Shape: (num_envs,)."""
    ee_tcp_pos = env.scene["ee_frame"].data.target_pos_w[..., 0, :]
    handle_pos = env.scene["drawer_frame"].data.target_pos_w[..., 0, :]
    return torch.norm(handle_pos - ee_tcp_pos, dim=-1, p=2)


def approach_ee_handle(
    env: ManagerBasedRLEnv,
    threshold: float = 0.12,
    max_distance: float = 0.28,
) -> torch.Tensor:
    """Inverse-square reward for EE approaching the drawer handle.

    Returns zero when EE–knob distance exceeds ``max_distance`` so home/idle poses
    cannot farm shaping reward. Inside ``threshold``, reward is doubled.
    """
    distance = _ee_handle_distance(env)
    reward = 1.0 / (1.0 + distance**2)
    reward = torch.pow(reward, 2)
    reward = torch.where(distance <= threshold, 2.0 * reward, reward)
    return torch.where(distance <= max_distance, reward, torch.zeros_like(reward))


def align_ee_handle(env: ManagerBasedRLEnv, max_distance: float = 0.28) -> torch.Tensor:
    """Reward for aligning EE orientation with the drawer handle.

    Gated by EE–knob distance so idle/home poses cannot farm orientation shaping alone.
    """
    ee_tcp_quat = env.scene["ee_frame"].data.target_quat_w[..., 0, :]
    handle_quat = env.scene["drawer_frame"].data.target_quat_w[..., 0, :]

    ee_mat = matrix_from_quat(ee_tcp_quat)
    handle_mat = matrix_from_quat(handle_quat)

    handle_x, handle_y = handle_mat[..., 0], handle_mat[..., 1]
    ee_x, ee_z = ee_mat[..., 0], ee_mat[..., 2]

    align_z = torch.bmm(ee_z.unsqueeze(1), -handle_x.unsqueeze(-1)).squeeze(-1).squeeze(-1)
    align_x = torch.bmm(ee_x.unsqueeze(1), -handle_y.unsqueeze(-1)).squeeze(-1).squeeze(-1)
    reward = 0.5 * (torch.sign(align_z) * align_z**2 + torch.sign(align_x) * align_x**2)
    distance = _ee_handle_distance(env)
    return torch.where(distance <= max_distance, reward, torch.zeros_like(reward))


def align_grasp_around_handle(
    env: ManagerBasedRLEnv,
    z_sigma: float = 0.015,
    y_min_sep: float = 0.008,
) -> torch.Tensor:
    """Reward coplanar pinch: fingers at handle height (Z) and on opposite Y sides."""
    lf_h, rf_h = _fingertips_in_handle_frame(env)

    z_err = 0.5 * (lf_h[:, 2].abs() + rf_h[:, 2].abs())
    height_rew = torch.exp(-((z_err / z_sigma) ** 2))

    y_straddle = lf_h[:, 1] * rf_h[:, 1] < 0.0
    y_sep = (lf_h[:, 1] - rf_h[:, 1]).abs()
    straddle_rew = y_straddle.float() * torch.tanh(y_sep / y_min_sep)

    return height_rew * straddle_rew


def approach_gripper_handle(
    env: ManagerBasedRLEnv,
    z_tol: float = 0.02,
    y_tol: float = 0.03,
    x_tol: float = 0.025,
) -> torch.Tensor:
    """Pull fingertips onto the knob: same Z height, straddle Y, reach handle depth X."""
    lf_h, rf_h = _fingertips_in_handle_frame(env)
    grasp_quality = align_grasp_around_handle(env)

    z_term = torch.clamp(z_tol - lf_h[:, 2].abs(), min=0.0) + torch.clamp(z_tol - rf_h[:, 2].abs(), min=0.0)
    x_term = torch.clamp(x_tol - lf_h[:, 0].abs(), min=0.0) + torch.clamp(x_tol - rf_h[:, 0].abs(), min=0.0)
    y_term = torch.clamp(y_tol - lf_h[:, 1].abs(), min=0.0) + torch.clamp(y_tol - rf_h[:, 1].abs(), min=0.0)

    return grasp_quality * (z_term + x_term + y_term)


def grasp_handle(
    env: ManagerBasedRLEnv,
    threshold: float,
    open_joint_pos: float,
    close_joint_pos: float,
    asset_cfg: SceneEntityCfg,
    grasp_align_threshold: float = 0.3,
) -> torch.Tensor:
    """Close gripper when EE TCP is near handle and pinch pose is ready.

    Distance gate uses EE TCP (same frame as policy ``rel_ee_handle`` obs).
    Pose gate uses ``align_grasp_around_handle`` (privileged sim shaping only).
    RH-P12: reward increases as rh_r1 moves from open (0 deg) toward closed (60 deg).
    """
    ee_tcp_pos = env.scene["ee_frame"].data.target_pos_w[..., 0, :]
    handle_pos = env.scene["drawer_frame"].data.target_pos_w[..., 0, :]
    gripper_joint_pos = _gripper_joint_pos(env, asset_cfg)

    distance = torch.norm(handle_pos - ee_tcp_pos, dim=-1, p=2)
    is_close = distance <= threshold
    is_ready = align_grasp_around_handle(env) >= grasp_align_threshold
    close_norm = _gripper_close_norm(gripper_joint_pos, open_joint_pos, close_joint_pos)

    return is_close * is_ready * close_norm


def open_drawer_bonus(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    gripper_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["rh_r1"]),
    close_threshold: float = math.radians(30.0),
) -> torch.Tensor:
    """Proportional bonus for drawer displacement (positive when drawer is open).

    Drawer joint: lowerLimit=-0.2 (fully open), upperLimit=0 (closed).
    We reward |pos| so the agent is motivated to pull the drawer.
    Stage-3 bonus requires a closed gripper (not pinch-only with open fingers).
    """
    drawer_pos = _joint_pos_scalar(env, asset_cfg)
    is_graspable = align_grasp_around_handle(env)
    is_closed = _gripper_is_closed(env, gripper_asset_cfg, close_threshold).float()
    return (is_graspable + 1.0) * is_closed * torch.abs(drawer_pos)


def multi_stage_open_drawer(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    gripper_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["rh_r1"]),
    close_threshold: float = math.radians(30.0),
) -> torch.Tensor:
    """Staged bonus at 1 cm, 10 cm, and 18 cm open.

    Thresholds use |drawer_pos| since joint goes negative when opening.
    Max travel is 0.2 m → stages at 0.01, 0.10, 0.18 m.
    Medium/hard stages require closed gripper + pinch-ready pose.
    """
    drawer_pos = torch.abs(_joint_pos_scalar(env, asset_cfg))
    is_graspable = align_grasp_around_handle(env)
    is_closed = _gripper_is_closed(env, gripper_asset_cfg, close_threshold).float()

    open_easy = (drawer_pos > 0.01).float() * 0.5
    open_medium = (drawer_pos > 0.10).float() * is_graspable * is_closed
    open_hard = (drawer_pos > 0.18).float() * is_graspable * is_closed
    return open_easy + open_medium + open_hard
