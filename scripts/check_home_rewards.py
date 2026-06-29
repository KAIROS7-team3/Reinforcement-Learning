"""Measure reward terms at a fixed arm home pose (no policy actions).

Compares legacy home vs current RL_HOME_JOINT_DEG after reset + joint teleport.

Usage:
    cd /home/user/Reinforcement-Learning
    /home/user/IsaacLab/isaaclab.sh -p scripts/check_home_rewards.py --headless
"""

from __future__ import annotations

import argparse
import math
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Isaac-OpenDrawer-Teacher-v0")
parser.add_argument("--num_envs", type=int, default=1)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "source"))
import tool_transfer_bot  # noqa: F401, E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from isaaclab.managers import SceneEntityCfg  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

from tool_transfer_bot.assets.doosan_e0509 import RL_HOME_JOINT_DEG  # noqa: E402
from tool_transfer_bot.tasks import mdp  # noqa: E402

LEGACY_HOME_DEG = {
    "joint_1": 0.0,
    "joint_2": 0.0,
    "joint_3": 90.0,
    "joint_4": 0.0,
    "joint_5": 90.0,
    "joint_6": 0.0,
    "rh_r1": 0.0,
    "rh_r2": 0.0,
    "rh_l1": 0.0,
    "rh_l2": 0.0,
}

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


def _eval_terms(raw) -> dict[str, float]:
    return {
        "approach_ee_handle": mdp.approach_ee_handle(raw)[0].item(),
        "align_ee_handle": mdp.align_ee_handle(raw)[0].item(),
        "approach_gripper_handle": mdp.approach_gripper_handle(raw)[0].item(),
        "align_grasp_around_handle": mdp.align_grasp_around_handle(raw)[0].item(),
        "grasp_handle": mdp.grasp_handle(
            raw,
            threshold=0.06,
            open_joint_pos=0.0,
            close_joint_pos=math.radians(60.0),
            asset_cfg=SceneEntityCfg("robot", joint_names=["rh_r1"]),
        )[0].item(),
        "open_drawer_bonus": mdp.open_drawer_bonus(
            raw,
            asset_cfg=SceneEntityCfg("toolbox", joint_names=["drawer_joint"]),
        )[0].item(),
        "multi_stage_open_drawer": mdp.multi_stage_open_drawer(
            raw,
            asset_cfg=SceneEntityCfg("toolbox", joint_names=["drawer_joint"]),
        )[0].item(),
    }


def _metrics(env, label: str) -> dict[str, float]:
    raw = env.unwrapped
    ee = raw.scene["ee_frame"].data.target_pos_w[0, 0]
    knob = raw.scene["drawer_frame"].data.target_pos_w[0, 0]
    dist = torch.norm(knob - ee).item()

    dt = raw.step_dt
    ep_s = raw.max_episode_length_s
    n_steps = int(ep_s / dt)
    terms = _eval_terms(raw)

    total_tb = 0.0
    approach_tb = 0.0
    print(f"\n=== {label} ===")
    print(f"  ee_handle_dist_m = {dist:.4f}")
    for name, raw_val in terms.items():
        w = REWARD_WEIGHTS[name]
        tb_r = raw_val * w * dt * n_steps / ep_s
        teleop_w = raw_val * w
        total_tb += tb_r
        if name == "approach_ee_handle":
            approach_tb = tb_r
        print(f"  {name:28s} raw={raw_val:8.5f}  teleop_w={teleop_w:7.4f}  TB_EpR={tb_r:7.4f}")
    print(f"  {'SUM (idle 8s TB-style)':28s} TB_total={total_tb:7.4f}")
    return {"dist_m": dist, "approach_tb": approach_tb, "total_tb": total_tb}


def main() -> None:
    print("[INFO] check_home_rewards starting", flush=True)
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env = gym.make(args_cli.task, cfg=env_cfg)
    env.reset()
    print("[INFO] env reset OK", flush=True)

    try:
        _set_robot_pose(env, LEGACY_HOME_DEG)
        print("[INFO] legacy pose set", flush=True)
        legacy = _metrics(env, "LEGACY home (0,0,90,0,90,0)")

        _set_robot_pose(env, RL_HOME_JOINT_DEG)
        print("[INFO] new pose set", flush=True)
        new = _metrics(env, "NEW RL_HOME_JOINT_DEG")
    except Exception:
        import traceback

        traceback.print_exc()
        raise

    print("\n=== Summary ===")
    print(f"  EE-knob distance: {legacy['dist_m']:.3f} m -> {new['dist_m']:.3f} m")
    print(f"  TB approach_ee_handle (idle): {legacy['approach_tb']:.3f} -> {new['approach_tb']:.3f}")
    print(f"  TB total reward terms (idle): {legacy['total_tb']:.3f} -> {new['total_tb']:.3f}")
    print("  Reference: teleop weighted approach_ee_handle >3.0 ≈ dist <= 0.1 m")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
