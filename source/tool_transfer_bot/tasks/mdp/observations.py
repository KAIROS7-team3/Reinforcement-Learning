"""Custom observation functions for tool-transfer tasks."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def ee_pos_w(env: ManagerBasedRLEnv) -> torch.Tensor:
    """EE TCP position in world frame. Shape: (num_envs, 3)."""
    return env.scene["ee_frame"].data.target_pos_w[..., 0, :]


def ee_quat_w(env: ManagerBasedRLEnv) -> torch.Tensor:
    """EE TCP orientation (quaternion w-first) in world frame. Shape: (num_envs, 4)."""
    return env.scene["ee_frame"].data.target_quat_w[..., 0, :]


def rel_ee_handle_distance(
    env: ManagerBasedRLEnv,
    ee_frame_cfg: SceneEntityCfg,
    handle_frame_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Vector from EE TCP to drawer handle in world frame. Shape: (num_envs, 3)."""
    ee_pos = env.scene[ee_frame_cfg.name].data.target_pos_w[..., 0, :]
    handle_pos = env.scene[handle_frame_cfg.name].data.target_pos_w[..., 0, :]
    return handle_pos - ee_pos


def target_tool_id(env: ManagerBasedRLEnv) -> torch.Tensor:
    """One-hot goal vector for the target tool. Shape: (num_envs, 6).

    The concrete task cfg must set ``env.target_tool_id_onehot`` as a
    (num_envs, 6) tensor before the first observation is collected.
    Falls back to all-zeros if the attribute is absent.
    """
    if hasattr(env, "target_tool_id_onehot"):
        return env.target_tool_id_onehot
    return torch.zeros(env.num_envs, 6, device=env.device)
