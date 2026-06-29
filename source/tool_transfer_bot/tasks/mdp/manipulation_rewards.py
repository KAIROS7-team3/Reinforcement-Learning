"""Pick-and-place rewards for FetchTool / ReturnTool.

Isaac Lab ``Lift``-style shaping (see ``isaaclab_tasks/.../lift/mdp/rewards.py``):
  1. EE → object reach (tanh)
  2. lift height binary bonus
  3. object → place target tracking (tanh, gated by lift height; coarse + fine std)
  4. terminal success bonus (ReturnTool termination alignment)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _object_and_target_pos(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg,
    target_frame_cfg: SceneEntityCfg,
) -> tuple[torch.Tensor, torch.Tensor]:
    obj = env.scene[object_cfg.name].data.root_pos_w
    tgt = env.scene[target_frame_cfg.name].data.target_pos_w[..., 0, :]
    return obj, tgt


def object_ee_distance(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("tool"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward reaching the object (tanh kernel on EE–object distance)."""
    obj = env.scene[object_cfg.name].data.root_pos_w
    ee = env.scene[ee_frame_cfg.name].data.target_pos_w[..., 0, :]
    distance = torch.linalg.norm(obj - ee, dim=-1)
    return 1.0 - torch.tanh(distance / max(std, 1e-4))


def object_is_lifted(
    env: ManagerBasedRLEnv,
    minimal_height: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("tool"),
) -> torch.Tensor:
    """Binary reward when object root Z exceeds ``minimal_height`` (world frame, m)."""
    obj_z = env.scene[object_cfg.name].data.root_pos_w[:, 2]
    return torch.where(obj_z > minimal_height, 1.0, 0.0)


def object_goal_tracking(
    env: ManagerBasedRLEnv,
    std: float,
    minimal_height: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("tool"),
    target_frame_cfg: SceneEntityCfg = SceneEntityCfg("target_frame"),
) -> torch.Tensor:
    """Track place target with tanh kernel, active only after lift (Isaac Lift gate)."""
    obj, tgt = _object_and_target_pos(env, object_cfg, target_frame_cfg)
    distance = torch.linalg.norm(obj - tgt, dim=-1)
    lifted = obj[:, 2] > minimal_height
    return lifted.float() * (1.0 - torch.tanh(distance / max(std, 1e-4)))


def success_bonus(
    env: ManagerBasedRLEnv,
    threshold: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("tool"),
    target_frame_cfg: SceneEntityCfg = SceneEntityCfg("target_frame"),
) -> torch.Tensor:
    """Binary bonus when object is within threshold of place target."""
    obj, tgt = _object_and_target_pos(env, object_cfg, target_frame_cfg)
    dist = torch.linalg.norm(obj - tgt, dim=-1)
    return (dist <= threshold).float()


# Backward-compatible alias used by older configs / scripts.
def ee_object_approach(
    env: ManagerBasedRLEnv,
    threshold: float = 0.1,
    max_distance: float | None = None,
    object_cfg: SceneEntityCfg = SceneEntityCfg("tool"),
) -> torch.Tensor:
    """Alias for ``object_ee_distance`` (``threshold`` maps to ``std``)."""
    del max_distance  # Lift style does not gate reach by max distance.
    return object_ee_distance(env, std=threshold, object_cfg=object_cfg)


def object_goal_dist(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("tool"),
    target_frame_cfg: SceneEntityCfg = SceneEntityCfg("target_frame"),
) -> torch.Tensor:
    """Raw L2 object–goal distance (utility / logging; use negative weight sparingly)."""
    obj, tgt = _object_and_target_pos(env, object_cfg, target_frame_cfg)
    return torch.linalg.norm(obj - tgt, dim=-1)
