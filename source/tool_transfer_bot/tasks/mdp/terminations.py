"""Custom termination functions for tool-transfer tasks."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import math as math_utils

from tool_transfer_bot.assets.environments import (
    RETURN_TOOL_DRAWER_FLOOR_X_MAX,
    RETURN_TOOL_DRAWER_FLOOR_X_MIN,
    RETURN_TOOL_DRAWER_FLOOR_Y_MAX,
    RETURN_TOOL_DRAWER_FLOOR_Y_MIN,
    RETURN_TOOL_DRAWER_TOOL_XY_MARGIN,
)

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
    asset = env.scene[asset_cfg.name]
    drawer_pos = asset.data.joint_pos[:, asset_cfg.joint_ids].squeeze(-1)
    return torch.abs(drawer_pos) >= threshold


def return_tool_success(
    env: ManagerBasedRLEnv,
    dist_threshold: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("tool"),
    target_frame_cfg: SceneEntityCfg = SceneEntityCfg("target_frame"),
) -> torch.Tensor:
    """ReturnTool success: object within dist_threshold of place target."""
    obj = env.scene[object_cfg.name].data.root_pos_w
    tgt = env.scene[target_frame_cfg.name].data.target_pos_w[..., 0, :]
    dist = torch.linalg.norm(obj - tgt, dim=-1)
    return dist <= dist_threshold


def _tool_pose_in_drawer_frame(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg,
    target_frame_cfg: SceneEntityCfg,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    tool = env.scene[object_cfg.name]
    frame = env.scene[target_frame_cfg.name]
    tool_pos = tool.data.root_pos_w
    tool_quat = tool.data.root_quat_w
    src_pos = frame.data.source_pos_w
    src_quat = frame.data.source_quat_w
    place_local = frame.data.target_pos_source[..., 0, :]
    tool_local, _ = math_utils.subtract_frame_transforms(
        src_pos, src_quat, tool_pos, tool_quat
    )
    return tool_local, place_local, tool.data.root_lin_vel_w


def demo_place_success_parts(
    env: ManagerBasedRLEnv,
    gripper_open_rad: float = 0.35,
    max_linear_vel: float = 0.10,
    drawer_x_min: float = RETURN_TOOL_DRAWER_FLOOR_X_MIN,
    drawer_x_max: float = RETURN_TOOL_DRAWER_FLOOR_X_MAX,
    drawer_y_min: float = RETURN_TOOL_DRAWER_FLOOR_Y_MIN,
    drawer_y_max: float = RETURN_TOOL_DRAWER_FLOOR_Y_MAX,
    tool_xy_margin: float = RETURN_TOOL_DRAWER_TOOL_XY_MARGIN,
    z_band: float = 0.04,
    z_center_local: float | None = None,
    object_cfg: SceneEntityCfg = SceneEntityCfg("tool"),
    target_frame_cfg: SceneEntityCfg = SceneEntityCfg("target_frame"),
) -> dict[str, torch.Tensor | float]:
    """Per-condition booleans for ``demo_place_success`` (env index 0 scalars for debug)."""
    tool_local, place_local, lin_vel = _tool_pose_in_drawer_frame(
        env, object_cfg, target_frame_cfg
    )
    robot = env.scene["robot"]

    grip_r1 = robot.data.joint_pos[:, robot.joint_names.index("rh_r1")]
    gripper_open = grip_r1 <= gripper_open_rad

    x_min = drawer_x_min + tool_xy_margin
    x_max = drawer_x_max - tool_xy_margin
    y_min = drawer_y_min + tool_xy_margin
    y_max = drawer_y_max - tool_xy_margin
    in_drawer_xy = (
        (tool_local[:, 0] >= x_min)
        & (tool_local[:, 0] <= x_max)
        & (tool_local[:, 1] >= y_min)
        & (tool_local[:, 1] <= y_max)
    )

    ref_z = (
        torch.full_like(tool_local[:, 2], z_center_local)
        if z_center_local is not None
        else place_local[:, 2]
    )
    dz = torch.abs(tool_local[:, 2] - ref_z)
    on_floor_z = dz <= z_band

    slow = torch.linalg.norm(lin_vel, dim=-1) <= max_linear_vel

    # Distances to place_target (marker 2) in drawer frame — for debug only.
    dx_marker = torch.abs(tool_local[:, 0] - place_local[:, 0])
    dy_marker = torch.abs(tool_local[:, 1] - place_local[:, 1])

    return {
        "gripper_open": gripper_open,
        "in_drawer_xy": in_drawer_xy,
        "on_floor_z": on_floor_z,
        "slow": slow,
        "grip_r1": grip_r1,
        "tool_local_x": tool_local[:, 0],
        "tool_local_y": tool_local[:, 1],
        "dx_marker": dx_marker,
        "dy_marker": dy_marker,
        "dz": dz,
        "speed": torch.linalg.norm(lin_vel, dim=-1),
        "drawer_x_min": x_min,
        "drawer_x_max": x_max,
        "drawer_y_min": y_min,
        "drawer_y_max": y_max,
    }


def demo_place_success(
    env: ManagerBasedRLEnv,
    gripper_open_rad: float = 0.35,
    max_linear_vel: float = 0.10,
    drawer_x_min: float = RETURN_TOOL_DRAWER_FLOOR_X_MIN,
    drawer_x_max: float = RETURN_TOOL_DRAWER_FLOOR_X_MAX,
    drawer_y_min: float = RETURN_TOOL_DRAWER_FLOOR_Y_MIN,
    drawer_y_max: float = RETURN_TOOL_DRAWER_FLOOR_Y_MAX,
    tool_xy_margin: float = RETURN_TOOL_DRAWER_TOOL_XY_MARGIN,
    z_band: float = 0.04,
    z_center_local: float | None = None,
    object_cfg: SceneEntityCfg = SceneEntityCfg("tool"),
    target_frame_cfg: SceneEntityCfg = SceneEntityCfg("target_frame"),
) -> torch.Tensor:
    """BC demo export: tool resting anywhere on open drawer floor.

    Position checks in drawer link frame (``target_frame`` source):
      - gripper open (rh_r1 <= gripper_open_rad)
      - tool root XY inside drawer floor AABB (minus tool_xy_margin for wall clearance)
      - tool root Z within z_band of drawer-floor center height (z_center_local or place_target z)
      - |linear_vel| <= max_linear_vel

    Distances use the rigid-body **root origin** (cube geometric center), not a corner.
    """
    parts = demo_place_success_parts(
        env,
        gripper_open_rad=gripper_open_rad,
        max_linear_vel=max_linear_vel,
        drawer_x_min=drawer_x_min,
        drawer_x_max=drawer_x_max,
        drawer_y_min=drawer_y_min,
        drawer_y_max=drawer_y_max,
        tool_xy_margin=tool_xy_margin,
        z_band=z_band,
        z_center_local=z_center_local,
        object_cfg=object_cfg,
        target_frame_cfg=target_frame_cfg,
    )
    return (
        parts["gripper_open"]
        & parts["in_drawer_xy"]
        & parts["on_floor_z"]
        & parts["slow"]
    )
