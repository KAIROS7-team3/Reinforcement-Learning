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
    """Vector from EE TCP to drawer knob frame in world frame. Shape: (num_envs, 3).

    ``handle_frame_cfg`` = ``drawer_frame`` (FurnitureKnob_01 center), not prim ``handle``.
    """
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


def object_pos_w(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("tool"),
) -> torch.Tensor:
    """Object root position relative to env origin. Shape: (num_envs, 3)."""
    asset = env.scene[asset_cfg.name]
    return asset.data.root_pos_w - env.scene.env_origins


def object_quat_w(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("tool"),
) -> torch.Tensor:
    """Object root orientation (w-first). Shape: (num_envs, 4)."""
    return env.scene[asset_cfg.name].data.root_quat_w


def target_pos_w(
    env: ManagerBasedRLEnv,
    target_frame_cfg: SceneEntityCfg = SceneEntityCfg("target_frame"),
) -> torch.Tensor:
    """Place target position in env frame. Shape: (num_envs, 3)."""
    pos = env.scene[target_frame_cfg.name].data.target_pos_w[..., 0, :]
    return pos - env.scene.env_origins


def target_quat_w(
    env: ManagerBasedRLEnv,
    target_frame_cfg: SceneEntityCfg = SceneEntityCfg("target_frame"),
) -> torch.Tensor:
    """Place target orientation (w-first). Shape: (num_envs, 4)."""
    return env.scene[target_frame_cfg.name].data.target_quat_w[..., 0, :]


def rel_ee_object_distance(
    env: ManagerBasedRLEnv,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("tool"),
) -> torch.Tensor:
    """Vector from EE TCP to object root. Shape: (num_envs, 3)."""
    ee = env.scene[ee_frame_cfg.name].data.target_pos_w[..., 0, :]
    obj = env.scene[object_cfg.name].data.root_pos_w
    return obj - ee
