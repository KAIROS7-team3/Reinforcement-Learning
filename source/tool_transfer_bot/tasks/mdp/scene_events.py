"""Scene-specific event helpers."""

from __future__ import annotations

import torch

from isaaclab.envs import ManagerBasedEnv

from tool_transfer_bot.assets.environments import TOOLBOX_BODY_WORLD_POS


def patch_toolbox_root_state(env: ManagerBasedEnv, env_ids: torch.Tensor | None = None) -> None:
    """Align PhysX articulation root with the visible toolbox body.

    Isaac Lab uses init_state.pos for both USD spawn (toolbox_with_handle root) and
    default_root_state (articulation root link). For this asset those differ because
    of nested xforms copied from with_camera.usda.
    """
    toolbox = env.scene["toolbox"]
    pos = torch.tensor(TOOLBOX_BODY_WORLD_POS, device=toolbox.device, dtype=torch.float32)
    toolbox.data.default_root_state[:, 0:3] = pos
