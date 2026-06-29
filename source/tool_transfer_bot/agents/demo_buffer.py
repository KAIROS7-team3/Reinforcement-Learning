"""Load Isaac Lab HDF5 demonstration datasets for BC training."""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset

from isaaclab.utils.datasets import HDF5DatasetFileHandler


@dataclass
class DemoBatch:
    obs: torch.Tensor
    actions: torch.Tensor


class DemoTransitionDataset(Dataset):
    """Flat (obs, action) pairs from all episodes in an HDF5 demo file."""

    def __init__(
        self,
        dataset_path: str,
        device: str = "cpu",
        *,
        exclude_episodes: frozenset[str] | None = None,
    ):
        if not os.path.isfile(dataset_path):
            raise FileNotFoundError(f"Demo dataset not found: {dataset_path}")

        handler = HDF5DatasetFileHandler()
        handler.open(dataset_path, mode="r")

        obs_list: list[torch.Tensor] = []
        act_list: list[torch.Tensor] = []
        skip = exclude_episodes or frozenset()

        for ep_name in handler.get_episode_names():
            if ep_name in skip:
                continue
            episode = handler.load_episode(ep_name, device=device)
            if episode is None:
                continue
            obs = episode.data.get("obs")
            actions = episode.data.get("actions")
            if obs is None or actions is None:
                continue
            # Recorder stores nested dict; use flat policy group if present.
            if isinstance(obs, dict):
                if "policy" in obs:
                    obs = obs["policy"]
                else:
                    obs = torch.cat([v.reshape(v.shape[0], -1) for v in obs.values()], dim=-1)
            if obs.ndim > 2:
                obs = obs.reshape(obs.shape[0], -1)
            if actions.ndim > 2:
                actions = actions.reshape(actions.shape[0], -1)
            # Teleop demos may record 10D (all gripper joints); BC/PPO use 7D.
            if actions.shape[-1] == 10:
                actions = actions[..., :7]
            n = min(obs.shape[0], actions.shape[0])
            if n == 0:
                continue
            obs_list.append(obs[:n].float())
            act_list.append(actions[:n].float())

        handler.close()

        if not obs_list:
            raise RuntimeError(f"No (obs, action) transitions found in {dataset_path}")

        self.obs = torch.cat(obs_list, dim=0)
        self.actions = torch.cat(act_list, dim=0)
        self.obs_dim = int(self.obs.shape[-1])
        self.action_dim = int(self.actions.shape[-1])

    def __len__(self) -> int:
        return self.obs.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.obs[idx], self.actions[idx]
