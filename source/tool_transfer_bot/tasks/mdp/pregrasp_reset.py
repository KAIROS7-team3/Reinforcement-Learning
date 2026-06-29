"""Reset robot arm to offline IK pre-grasp waypoints (Method A)."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def reset_robot_to_pregrasp(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    joint_positions_deg: dict[str, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    position_range: tuple[float, float] = (0.0, 0.0),
    velocity_range: tuple[float, float] = (0.0, 0.0),
):
    """Teleport robot joints to exported pre-grasp pose with optional DR noise on arm."""
    asset: Articulation = env.scene[asset_cfg.name]

    joint_pos = asset.data.default_joint_pos[env_ids].clone()
    joint_vel = torch.zeros_like(joint_pos)

    for name, value_deg in joint_positions_deg.items():
        if name not in asset.joint_names:
            continue
        idx = asset.joint_names.index(name)
        value = math.radians(value_deg) if name.startswith("joint_") else value_deg
        joint_pos[:, idx] = value

    if position_range != (0.0, 0.0):
        arm_indices = [
            asset.joint_names.index(n)
            for n in joint_positions_deg
            if n.startswith("joint_") and n in asset.joint_names
        ]
        noise = math_utils.sample_uniform(*position_range, joint_pos.shape, joint_pos.device)
        for idx in arm_indices:
            joint_pos[:, idx] += noise[:, idx]

    if velocity_range != (0.0, 0.0):
        joint_vel += math_utils.sample_uniform(*velocity_range, joint_vel.shape, joint_vel.device)

    limits = asset.data.soft_joint_pos_limits[env_ids]
    joint_pos = joint_pos.clamp_(limits[..., 0], limits[..., 1])
    vel_limits = asset.data.soft_joint_vel_limits[env_ids]
    joint_vel = joint_vel.clamp_(-vel_limits, vel_limits)

    asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
    asset.set_joint_position_target(joint_pos, env_ids=env_ids)
    asset.set_joint_velocity_target(joint_vel, env_ids=env_ids)
