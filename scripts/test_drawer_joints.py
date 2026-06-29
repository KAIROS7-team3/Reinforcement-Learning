"""Animate toolbox drawer joints and print positions (physics sanity check).

Usage:
    cd /home/user/IsaacLab
    ./isaaclab.sh -p /home/user/Reinforcement-Learning/scripts/test_drawer_joints.py

What to look for:
  - drawer_joint moves smoothly (closed=0 → open=-0.2 m on Y prismatic)
  - Visual mesh follows physics (no sliding / stuck drawer)
  - Limits clamp at 0 and -0.2
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Test toolbox drawer joint motion in RL scene.")
parser.add_argument("--task", type=str, default="Isaac-OpenDrawer-Teacher-Play-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--cycles", type=int, default=3, help="Open-close cycles per drawer.")
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

DRAWER_JOINTS = ("drawer_joint",)
OPEN_POS = -0.2  # m (with_camera / USD lower limit)
STEPS_PER_HALF = 120  # sim steps for open or close sweep


def _joint_limits(toolbox) -> None:
    names = toolbox.joint_names
    lo = toolbox.data.soft_joint_pos_limits[0, :, 0]
    hi = toolbox.data.soft_joint_pos_limits[0, :, 1]
    print("[INFO] Toolbox joints:")
    for i, name in enumerate(names):
        print(f"  {name}: soft limits [{lo[i]:.4f}, {hi[i]:.4f}]")


def _print_positions(toolbox, label: str) -> None:
    pos = toolbox.data.joint_pos[0].cpu()
    parts = []
    for name in DRAWER_JOINTS:
        idx = toolbox.joint_names.index(name)
        parts.append(f"{name}={pos[idx]:.4f}")
    print(f"  {label}: " + ", ".join(parts))


def main():
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    # task_success fires at |drawer_joint| >= 0.15 m and resets drawer to closed.
    env_cfg.terminations.task_success = None
    env = gym.make(args_cli.task, cfg=env_cfg)
    env.reset()

    toolbox = env.unwrapped.scene["toolbox"]
    device = toolbox.device
    _joint_limits(toolbox)

    drawer_ids = torch.tensor(
        [toolbox.joint_names.index(n) for n in DRAWER_JOINTS],
        device=device,
        dtype=torch.long,
    )

    zero_actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)

    for cycle in range(args_cli.cycles):
        for joint_name in DRAWER_JOINTS:
            j_id = toolbox.joint_names.index(joint_name)
            print(f"\n[INFO] Cycle {cycle + 1}/{args_cli.cycles} — driving {joint_name}")

            # open: 0 → -0.2
            for step in range(STEPS_PER_HALF):
                t = step / max(STEPS_PER_HALF - 1, 1)
                target = OPEN_POS * t
                q = toolbox.data.joint_pos.clone()
                q[:, j_id] = target
                toolbox.write_joint_position_to_sim(q)
                toolbox.write_joint_velocity_to_sim(torch.zeros_like(q))
                env.step(zero_actions)
                if step % 30 == 0 or step == STEPS_PER_HALF - 1:
                    _print_positions(toolbox, f"open step {step}")

            # close: -0.2 → 0
            for step in range(STEPS_PER_HALF):
                t = step / max(STEPS_PER_HALF - 1, 1)
                target = OPEN_POS * (1.0 - t)
                q = toolbox.data.joint_pos.clone()
                q[:, j_id] = target
                toolbox.write_joint_position_to_sim(q)
                toolbox.write_joint_velocity_to_sim(torch.zeros_like(q))
                env.step(zero_actions)
                if step % 30 == 0 or step == STEPS_PER_HALF - 1:
                    _print_positions(toolbox, f"close step {step}")

    print("\n[INFO] Done. If drawers did not move visually, check articulation root / collision setup.")
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
