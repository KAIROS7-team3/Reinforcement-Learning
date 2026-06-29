"""Run trained policy and log drawer joint positions while the robot interacts.

Use this to verify that drawer_joint / drawer_02_joint change when the gripper
pulls the handle (not when joints are driven manually).

Usage:
    cd /home/user/IsaacLab
    ./isaaclab.sh -p /home/user/Reinforcement-Learning/scripts/monitor_drawer_play.py \\
        --task Isaac-OpenDrawer-Teacher-Play-v0 \\
        --num_envs 1 \\
        --load_run 2026-06-12_00-46-52 \\
        --checkpoint model_900.pt \\
        --episodes 3

Or absolute checkpoint path:
    ./isaaclab.sh -p .../monitor_drawer_play.py \\
        --checkpoint /home/user/IsaacLab/logs/rsl_rl/open_drawer_teacher/.../model_900.pt
"""

import argparse
import csv
import os
import sys
from isaaclab.app import AppLauncher

# Isaac Lab rsl_rl CLI (--load_run, --checkpoint, etc.)
_ISAACLAB_RSL_RL = os.path.join(
    os.environ.get("ISAACLAB_PATH", "/home/user/IsaacLab"),
    "scripts",
    "reinforcement_learning",
    "rsl_rl",
)
sys.path.insert(0, _ISAACLAB_RSL_RL)
import cli_args  # noqa: E402

parser = argparse.ArgumentParser(description="Monitor drawer joints during policy play.")
parser.add_argument("--task", type=str, default="Isaac-OpenDrawer-Teacher-Play-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--episodes", type=int, default=3, help="Number of episodes to run.")
parser.add_argument("--log_interval", type=int, default=20, help="Print every N sim steps.")
parser.add_argument(
    "--csv",
    type=str,
    default="/tmp/drawer_joint_monitor.csv",
    help="CSV log path (step, drawer_joint, drawer_02_joint, gripper_rh_r1, ee_handle_dist).",
)
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O."
)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import importlib.metadata as metadata  # noqa: E402
import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from packaging import version  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent  # noqa: E402
from isaaclab.utils.assets import retrieve_file_path  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import get_checkpoint_path, load_cfg_from_registry, parse_env_cfg  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "source"))
import tool_transfer_bot  # noqa: F401, E402

DRAWER_JOINTS = ("drawer_joint",)
OPEN_THRESHOLD_M = 0.05  # |position| above this counts as "opening"


def _resolve_checkpoint(agent_cfg) -> str:
    if args_cli.checkpoint:
        return retrieve_file_path(args_cli.checkpoint)
    log_root = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    return get_checkpoint_path(log_root, agent_cfg.load_run, agent_cfg.load_checkpoint)


def _drawer_state(toolbox, robot) -> dict[str, float]:
    pos = toolbox.data.joint_pos[0].cpu()
    out = {}
    for name in DRAWER_JOINTS:
        out[name] = float(pos[toolbox.joint_names.index(name)])
    if "rh_r1" in robot.joint_names:
        gpos = robot.data.joint_pos[0].cpu()
        out["gripper_rh_r1"] = float(gpos[robot.joint_names.index("rh_r1")])
    return out


def _ee_handle_distance_m(env) -> float:
    ee = env.unwrapped.scene["ee_frame"].data.target_pos_w[0, 0]
    handle = env.unwrapped.scene["drawer_frame"].data.target_pos_w[0, 0]
    return float(torch.linalg.norm(ee - handle).item())


def main():
    task_name = args_cli.task.split(":")[-1]
    env_cfg = parse_env_cfg(
        task_name,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    agent_cfg = load_cfg_from_registry(task_name, "rsl_rl_cfg_entry_point")
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)

    resume_path = _resolve_checkpoint(agent_cfg)
    agent_yaml = os.path.join(os.path.dirname(resume_path), "params", "agent.yaml")
    if os.path.isfile(agent_yaml):
        import yaml

        with open(agent_yaml, encoding="utf-8") as f:
            saved = yaml.safe_load(f)
        saved_std = (saved.get("actor", {}).get("distribution_cfg") or {}).get("std_type")
        if saved_std and hasattr(agent_cfg, "policy"):
            agent_cfg.policy.noise_std_type = saved_std

    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))
    print(f"[INFO] Checkpoint: {resume_path}")

    env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    rsl_version = version.parse(metadata.version("rsl-rl-lib"))
    if rsl_version >= version.parse("4.0.0"):
        policy_nn = None
    elif rsl_version >= version.parse("2.3.0"):
        policy_nn = runner.alg.policy
    else:
        policy_nn = runner.alg.actor_critic

    raw_env = env.unwrapped
    toolbox = raw_env.scene["toolbox"]
    robot = raw_env.scene["robot"]
    max_episode_steps = int(raw_env.max_episode_length)

    rows: list[dict] = []
    global_step = 0

    with open(args_cli.csv, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "episode",
                "step",
                "drawer_joint",
                "gripper_rh_r1",
                "ee_handle_dist_m",
                "opened",
            ],
        )
        writer.writeheader()

        for ep in range(args_cli.episodes):
            obs = env.get_observations()
            ep_start = _drawer_state(toolbox, robot)
            ep_max = {k: abs(v) for k, v in ep_start.items() if k in DRAWER_JOINTS}
            ep_steps = 0
            opened_flag = False

            print(f"\n=== Episode {ep + 1}/{args_cli.episodes} ===")
            print(f"  start: drawer_joint={ep_start['drawer_joint']:.4f}")

            while ep_steps < max_episode_steps:
                with torch.inference_mode():
                    actions = policy(obs)
                    obs, _, dones, _ = env.step(actions)
                    if rsl_version >= version.parse("4.0.0"):
                        policy.reset(dones)
                    else:
                        policy_nn.reset(dones)

                state = _drawer_state(toolbox, robot)
                dist = _ee_handle_distance_m(raw_env)
                opened = abs(state["drawer_joint"]) >= OPEN_THRESHOLD_M
                opened_flag = opened_flag or opened
                for name in DRAWER_JOINTS:
                    ep_max[name] = max(ep_max[name], abs(state[name]))

                row = {
                    "episode": ep + 1,
                    "step": global_step,
                    "drawer_joint": state["drawer_joint"],
                    "gripper_rh_r1": state.get("gripper_rh_r1", 0.0),
                    "ee_handle_dist_m": dist,
                    "opened": int(opened),
                }
                writer.writerow(row)
                rows.append(row)

                if global_step % args_cli.log_interval == 0:
                    print(
                        f"  step {ep_steps:4d} | drawer_joint={state['drawer_joint']:+.4f} | "
                        f"gripper={state.get('gripper_rh_r1', 0.0):.3f} | "
                        f"ee-handle={dist:.3f} m"
                    )

                global_step += 1
                ep_steps += 1

                if dones.any():
                    break

            print(
                f"  end summary | max |drawer_joint|={ep_max['drawer_joint']:.4f} | "
                f"opened(>{OPEN_THRESHOLD_M}m)={opened_flag}"
            )

    print(f"\n[INFO] CSV written to {args_cli.csv}")
    print("[INFO] drawer_joint should move toward -0.2 m when the drawer opens (closed=0).")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
