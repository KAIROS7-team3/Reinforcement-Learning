"""Set goal-conditioning one-hot on env reset."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

NUM_TOOLS = 6


def set_target_tool_id_onehot(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    tool_index: int,
) -> None:
    """Store fixed one-hot target tool id on ``env`` (used by ``target_tool_id`` obs)."""
    if not hasattr(env, "target_tool_id_onehot"):
        env.target_tool_id_onehot = torch.zeros(env.num_envs, NUM_TOOLS, device=env.device)
    env.target_tool_id_onehot[env_ids, :] = 0.0
    env.target_tool_id_onehot[env_ids, tool_index] = 1.0
