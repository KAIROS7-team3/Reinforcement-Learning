"""Print robot arm joint positions (deg) after env reset — verify home pose.

Usage:
    cd /home/user/IsaacLab
    ./isaaclab.sh -p /home/user/Reinforcement-Learning/scripts/check_robot_home.py --headless
"""

import argparse
import math
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Isaac-OpenDrawer-Teacher-Play-v0")
parser.add_argument("--num_envs", type=int, default=1)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args(["--headless"])

app = AppLauncher(args).app

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "source"))
import tool_transfer_bot  # noqa: F401, E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

from tool_transfer_bot.assets.doosan_e0509 import RL_HOME_JOINT_DEG  # noqa: E402

EXPECTED_DEG = dict(RL_HOME_JOINT_DEG)

env_cfg = parse_env_cfg(args.task, device="cuda:0", num_envs=args.num_envs)
env = gym.make(args.task, cfg=env_cfg)
env.reset()

robot = env.unwrapped.scene["robot"]
default = robot.data.default_joint_pos[0].cpu()
actual = robot.data.joint_pos[0].cpu()

print(f"[INFO] task={args.task}")
print("[INFO] default_joint_pos from cfg (degrees) — training home pose:")
for name in ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "rh_r1"]:
    if name not in robot.joint_names:
        continue
    idx = robot.joint_names.index(name)
    deg = math.degrees(float(default[idx]))
    exp = EXPECTED_DEG.get(name)
    tag = f" (target {exp})" if exp is not None else ""
    print(f"  {name}: {deg:+.2f} deg{tag}")

print("[INFO] joint_pos after reset (degrees) — includes ±0.05 rad DR noise on arm:")
for name in ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]:
    idx = robot.joint_names.index(name)
    deg = math.degrees(float(actual[idx]))
    print(f"  {name}: {deg:+.2f} deg")

env.close()
app.close()
