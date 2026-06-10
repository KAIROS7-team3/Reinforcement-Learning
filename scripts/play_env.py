"""Visualise the RL scene with zero actions (no training).

Usage (inside Isaac Lab Docker):
    ./isaaclab.sh -p /home/user/Reinforcement-Learning/scripts/play_env.py \\
        --task Isaac-OpenDrawer-Teacher-Play-v0 --num_envs 1
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Play Isaac Lab tool-transfer environments.")
parser.add_argument("--task", type=str, default="Isaac-OpenDrawer-Teacher-Play-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O."
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "source"))
import tool_transfer_bot  # noqa: F401, E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402


def main():
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    env = gym.make(args_cli.task, cfg=env_cfg)
    print(f"[INFO] task={args_cli.task}  obs={env.observation_space}  act={env.action_space}")

    env.reset()
    while simulation_app.is_running():
        with torch.inference_mode():
            actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
            env.step(actions)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
