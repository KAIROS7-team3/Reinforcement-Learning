"""Compare reward terms at home vs exported pre-grasp waypoint.

Usage:
    export CONDA_PREFIX=/home/user/miniconda3/envs/env_isaaclab
    /home/user/IsaacLab/isaaclab.sh -p scripts/check_pregrasp_rewards.py --headless
"""

from __future__ import annotations

import argparse
import importlib
import math
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Isaac-OpenDrawer-Teacher-v0")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "source"))
import tool_transfer_bot  # noqa: F401, E402
import tool_transfer_bot.assets.pregrasp_waypoints as pregrasp_mod  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from isaaclab.managers import SceneEntityCfg  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

from tool_transfer_bot.assets.doosan_e0509 import RL_HOME_JOINT_DEG  # noqa: E402
from tool_transfer_bot.tasks import mdp  # noqa: E402

importlib.reload(pregrasp_mod)
PREGRASP = pregrasp_mod.OPEN_DRAWER_PREGRASP_JOINT_DEG
META = pregrasp_mod.OPEN_DRAWER_PREGRASP_META

REWARD_WEIGHTS = {
    "approach_ee_handle": 2.0,
    "align_ee_handle": 0.5,
    "approach_gripper_handle": 5.0,
    "align_grasp_around_handle": 0.5,
    "grasp_handle": 0.5,
    "open_drawer_bonus": 7.5,
    "multi_stage_open_drawer": 1.0,
}


def _set_robot_pose(env, home_deg: dict[str, float]) -> None:
    robot = env.unwrapped.scene["robot"]
    raw = env.unwrapped
    pos = robot.data.default_joint_pos.clone()
    for name, deg in home_deg.items():
        if name not in robot.joint_names:
            continue
        idx = robot.joint_names.index(name)
        pos[:, idx] = math.radians(deg) if name.startswith("joint_") else deg
    zero = torch.zeros_like(pos)
    robot.write_joint_state_to_sim(pos, zero)
    robot.set_joint_position_target(pos)
    raw.scene.write_data_to_sim()
    raw.sim.step(render=False)
    raw.scene.update(dt=raw.physics_dt)


def _metrics(env, label: str) -> dict[str, float]:
    raw = env.unwrapped
    ee = raw.scene["ee_frame"].data.target_pos_w[0, 0]
    knob = raw.scene["drawer_frame"].data.target_pos_w[0, 0]
    dist = torch.norm(knob - ee).item()

    terms = {
        "approach_ee_handle": mdp.approach_ee_handle(raw)[0].item(),
        "align_ee_handle": mdp.align_ee_handle(raw)[0].item(),
        "approach_gripper_handle": mdp.approach_gripper_handle(raw)[0].item(),
    }

    print(f"\n=== {label} ===")
    print(f"  ee_handle_dist_m = {dist:.4f}")
    approach_tb = 0.0
    for name, raw_val in terms.items():
        w = REWARD_WEIGHTS[name]
        tb_r = raw_val * w
        if name == "approach_ee_handle":
            approach_tb = tb_r
        print(f"  {name:28s} raw={raw_val:8.5f}  weighted={tb_r:7.4f}")
    return {"dist_m": dist, "approach_tb": approach_tb}


def main() -> None:
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    env = gym.make(args_cli.task, cfg=env_cfg)
    env.reset()

    _set_robot_pose(env, RL_HOME_JOINT_DEG)
    home = _metrics(env, "HOME")

    _set_robot_pose(env, PREGRASP)
    pre = _metrics(env, "PRE-GRASP")

    print("\n=== Meta ===")
    for k, v in META.items():
        if k != "probe":
            print(f"  {k}: {v}")

    print(f"\n=== Summary ===")
    print(f"  EE-knob: {home['dist_m']:.3f} m -> {pre['dist_m']:.3f} m")
    print(f"  approach weighted: {home['approach_tb']:.3f} -> {pre['approach_tb']:.3f}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
