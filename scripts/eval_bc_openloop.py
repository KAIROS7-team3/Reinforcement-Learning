#!/usr/bin/env python3
"""Open-loop BC eval: policy(HDF5 obs) vs demo actions (no sim closed-loop).

Usage:
    ../IsaacLab/isaaclab.sh -p scripts/eval_bc_openloop.py \\
        --checkpoint data/checkpoints/return_tool/bc_warmstart.pt \\
        --dataset ./data/demos/return_tool/dataset.hdf5 \\
        --headless
"""

from __future__ import annotations

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Open-loop BC MSE on demo HDF5 obs.")
parser.add_argument("--task", type=str, default="Isaac-ReturnTool-Teacher-v0")
parser.add_argument("--dataset", type=str, required=True)
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument(
    "--grasp_phase_start",
    type=int,
    default=25,
    help="Report max|a-demo| from this frame index (grasp approach).",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "source"))
import tool_transfer_bot  # noqa: F401, E402

import importlib.metadata as metadata  # noqa: E402
import gymnasium as gym  # noqa: E402
import h5py  # noqa: E402
import torch  # noqa: E402
from packaging import version  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402
from tensordict import TensorDict  # noqa: E402

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent  # noqa: E402
from isaaclab.utils.assets import retrieve_file_path  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry, parse_env_cfg  # noqa: E402

POLICY_OBS_GROUP = "policy"


def _obs_batch(obs: torch.Tensor, device: torch.device) -> TensorDict | torch.Tensor:
    obs = obs.to(device)
    if version.parse(metadata.version("rsl-rl-lib")) >= version.parse("4.0.0"):
        return TensorDict({POLICY_OBS_GROUP: obs}, batch_size=[obs.shape[0]], device=device)
    return obs


def main() -> None:
    device = torch.device(args_cli.device if args_cli.device else "cuda:0")
    task_name = args_cli.task.split(":")[-1]
    env_cfg = parse_env_cfg(task_name, device=str(device), num_envs=1)
    agent_cfg = load_cfg_from_registry(task_name, "rsl_rl_cfg_entry_point")
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))

    env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    ckpt = retrieve_file_path(args_cli.checkpoint)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(ckpt)
    policy = runner.get_inference_policy(device=device)

    path = os.path.abspath(args_cli.dataset)
    all_mse: list[float] = []
    grasp_max: list[float] = []

    with h5py.File(path, "r") as f:
        eps = sorted(k for k in f["data"].keys() if k.startswith("demo_"))
        for ep in eps:
            obs = torch.tensor(f["data"][ep]["obs"][:], dtype=torch.float32)
            act = torch.tensor(f["data"][ep]["actions"][:], dtype=torch.float32)
            n = min(len(obs), len(act))
            if n == 0:
                continue
            with torch.inference_mode():
                pred = policy(_obs_batch(obs[:n], device))
            if pred.dim() == 1:
                pred = pred.unsqueeze(0)
            err = (pred.cpu() - act[:n]).abs()
            mse = float((pred.cpu() - act[:n]).pow(2).mean().item())
            all_mse.append(mse)
            start = min(args_cli.grasp_phase_start, n)
            grasp_max.append(float(err[start:].max().item()))

    env.close()

    print(f"[INFO] Open-loop BC on {len(all_mse)} episodes (no sim feedback)")
    print(f"  all-frame MSE: mean={sum(all_mse)/len(all_mse):.6f} max={max(all_mse):.6f}")
    print(
        f"  grasp phase (frame>={args_cli.grasp_phase_start}) max|a-demo|: "
        f"mean={sum(grasp_max)/len(grasp_max):.3f} max={max(grasp_max):.3f} rad",
    )
    print("  gate: grasp max|a-demo| < 0.1 rad → strong open-loop; >0.5 → weak at grasp phase")


if __name__ == "__main__":
    main()
    simulation_app.close()
