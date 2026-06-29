"""Batch-eval a trained open-drawer checkpoint (success rate, drawer travel, gripper).

Usage:
    cd /home/user/Reinforcement-Learning
    bash /home/user/IsaacLab/isaaclab.sh -p scripts/eval_policy_batch.py \\
        --headless \\
        --task Isaac-OpenDrawer-Teacher-Play-v0 \\
        --checkpoint logs/rsl_rl/open_drawer_teacher/2026-06-19_17-42-46_pregrasp_v1/model_100.pt \\
        --num_envs 16 \\
        --episodes 64
"""

from __future__ import annotations

import argparse
import os
import sys

from isaaclab.app import AppLauncher

_ISAACLAB_RSL_RL = os.path.join(
    os.environ.get("ISAACLAB_PATH", "/home/user/IsaacLab"),
    "scripts",
    "reinforcement_learning",
    "rsl_rl",
)
sys.path.insert(0, _ISAACLAB_RSL_RL)
import cli_args  # noqa: E402

parser = argparse.ArgumentParser(description="Batch policy evaluation for open-drawer task.")
parser.add_argument("--task", type=str, default="Isaac-OpenDrawer-Teacher-Play-v0")
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--episodes", type=int, default=64, help="Total episodes across all envs.")
parser.add_argument("--success_threshold_m", type=float, default=0.15)
parser.add_argument("--disable_fabric", action="store_true", default=False)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import importlib.metadata as metadata  # noqa: E402
import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from packaging import version  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent  # noqa: E402
from isaaclab.utils.assets import retrieve_file_path  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import get_checkpoint_path, load_cfg_from_registry, parse_env_cfg  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "source"))
import tool_transfer_bot  # noqa: F401, E402


def _resolve_checkpoint(agent_cfg) -> str:
    if args_cli.checkpoint:
        return retrieve_file_path(args_cli.checkpoint)
    log_root = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    return get_checkpoint_path(log_root, agent_cfg.load_run, agent_cfg.load_checkpoint)


def _load_saved_std_type(checkpoint_path: str) -> str | None:
    run_dir = os.path.dirname(checkpoint_path)
    agent_yaml = os.path.join(run_dir, "params", "agent.yaml")
    if not os.path.isfile(agent_yaml):
        return None
    with open(agent_yaml, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    dist = data.get("actor", {}).get("distribution_cfg") or {}
    return dist.get("std_type")


def main() -> None:
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
    saved_std = _load_saved_std_type(resume_path)
    if saved_std and hasattr(agent_cfg, "policy"):
        agent_cfg.policy.noise_std_type = saved_std
        print(f"[INFO] Matched checkpoint noise_std_type={saved_std}")

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
    policy_nn = None if rsl_version >= version.parse("4.0.0") else (
        runner.alg.policy if rsl_version >= version.parse("2.3.0") else runner.alg.actor_critic
    )

    raw = env.unwrapped
    toolbox = raw.scene["toolbox"]
    robot = raw.scene["robot"]
    drawer_idx = toolbox.joint_names.index("drawer_joint")
    gripper_idx = robot.joint_names.index("rh_r1") if "rh_r1" in robot.joint_names else None

    max_steps = int(raw.max_episode_length)
    target_eps = args_cli.episodes
    completed = 0
    successes = 0
    partial_5cm = 0
    max_travel: list[float] = []
    min_ee_dist: list[float] = []
    max_gripper_close: list[float] = []

    obs = env.get_observations()
    ep_steps = torch.zeros(args_cli.num_envs, dtype=torch.int, device=raw.device)
    ep_max_drawer = torch.zeros(args_cli.num_envs, device=raw.device)
    ep_min_dist = torch.full((args_cli.num_envs,), float("inf"), device=raw.device)
    ep_max_grip = torch.zeros(args_cli.num_envs, device=raw.device)

    print(f"[INFO] Evaluating {target_eps} episodes ({args_cli.num_envs} parallel envs)...")

    while completed < target_eps:
        with torch.inference_mode():
            actions = policy(obs)
            obs, rew, dones, infos = env.step(actions)
            if rsl_version >= version.parse("4.0.0"):
                policy.reset(dones)
            else:
                policy_nn.reset(dones)

        drawer = torch.abs(toolbox.data.joint_pos[:, drawer_idx])
        ee = raw.scene["ee_frame"].data.target_pos_w[:, 0]
        knob = raw.scene["drawer_frame"].data.target_pos_w[:, 0]
        dist = torch.linalg.norm(knob - ee, dim=-1)

        ep_max_drawer = torch.maximum(ep_max_drawer, drawer)
        ep_min_dist = torch.minimum(ep_min_dist, dist)
        if gripper_idx is not None:
            ep_max_grip = torch.maximum(ep_max_grip, robot.data.joint_pos[:, gripper_idx])
        ep_steps += 1

        done_mask = torch.as_tensor(dones, device=raw.device).view(-1).bool() | (ep_steps >= max_steps)
        if not done_mask.any():
            continue

        done_ids = done_mask.nonzero(as_tuple=False).flatten()
        for idx in done_ids.tolist():
            if completed >= target_eps:
                break
            travel = float(ep_max_drawer[idx].item())
            max_travel.append(travel)
            min_ee_dist.append(float(ep_min_dist[idx].item()))
            if gripper_idx is not None:
                max_gripper_close.append(float(ep_max_grip[idx].item()))
            if travel >= args_cli.success_threshold_m:
                successes += 1
            if travel >= 0.05:
                partial_5cm += 1
            completed += 1

        ep_max_drawer[done_ids] = 0.0
        ep_min_dist[done_ids] = float("inf")
        ep_max_grip[done_ids] = 0.0
        ep_steps[done_ids] = 0

    def _mean(xs: list[float]) -> float:
        return sum(xs) / len(xs) if xs else 0.0

    print("\n========== EVAL SUMMARY ==========")
    print(f"Checkpoint     : {resume_path}")
    print(f"Episodes       : {completed}")
    print(f"Success (>={args_cli.success_threshold_m:.2f} m): {successes}/{completed} ({100.0 * successes / completed:.1f}%)")
    print(f"Partial (>0.05 m) : {partial_5cm}/{completed} ({100.0 * partial_5cm / completed:.1f}%)")
    print(f"Mean max drawer travel : {_mean(max_travel):.4f} m")
    print(f"Max drawer travel      : {max(max_travel) if max_travel else 0.0:.4f} m")
    print(f"Mean min EE-knob dist  : {_mean(min_ee_dist):.4f} m")
    if max_gripper_close:
        print(f"Mean max gripper rh_r1 : {_mean(max_gripper_close):.3f} rad ({_mean(max_gripper_close) * 57.3:.1f} deg)")
    print("==================================")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
