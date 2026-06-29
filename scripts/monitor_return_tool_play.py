"""Run ReturnTool BC/PPO policy and log joint targets vs achieved positions.

Usage:
    cd /home/user/Reinforcement-Learning
    ../IsaacLab/isaaclab.sh -p scripts/monitor_return_tool_play.py \\
        --task Isaac-ReturnTool-Teacher-Play-v0 \\
        --checkpoint logs/rsl_rl/return_tool_teacher/bc_warmstart/model_0.pt \\
        --kinematic --interp_substeps 8 \\
        --warmup_steps 32 --warmup_demo demo_0 \\
        --compare_demo demo_0 --steps 40 --log_interval 1
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

parser = argparse.ArgumentParser(description="Monitor ReturnTool policy vs joint motion.")
parser.add_argument("--task", type=str, default="Isaac-ReturnTool-Teacher-Play-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--steps", type=int, default=300, help="Policy/open-loop steps after warmup.")
parser.add_argument("--log_interval", type=int, default=30, help="Print every N steps.")
parser.add_argument(
    "--dataset",
    type=str,
    default="./data/demos/return_tool/dataset.hdf5",
    help="HDF5 for demo replay / comparison.",
)
parser.add_argument(
    "--open_loop_demo",
    type=str,
    default=None,
    help="Replay actions from this demo key instead of policy.",
)
parser.add_argument(
    "--warmup_demo",
    type=str,
    default="demo_0",
    help="Demo key for pre-policy open-loop warmup.",
)
parser.add_argument(
    "--warmup_steps",
    type=int,
    default=0,
    help="Open-loop kinematic steps before policy (e.g. 32 = pre-grasp hold).",
)
parser.add_argument(
    "--compare_demo",
    type=str,
    default=None,
    help="If set, log |policy_action - demo_action| each step (e.g. demo_0).",
)
parser.add_argument(
    "--interp_substeps",
    type=int,
    default=1,
    help="Joint linear interp per step when --kinematic (1=teleport, 8=smoother).",
)
parser.add_argument(
    "--kinematic",
    action="store_true",
    default=False,
    help="Sketch replay: advance_bc_sketch_timestep (no PhysX PD). Use for BC checkpoints.",
)
parser.add_argument(
    "--no_grasp_assist",
    action="store_true",
    default=False,
    help="Disable table-lock/grasp weld (demo collection uses assist ON).",
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
import h5py  # noqa: E402
import torch  # noqa: E402
from packaging import version  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent  # noqa: E402
from isaaclab.utils.assets import retrieve_file_path  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import get_checkpoint_path, load_cfg_from_registry, parse_env_cfg  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "source"))
import tool_transfer_bot  # noqa: F401, E402
from tool_transfer_bot.tasks.mdp.grasp_assist import GraspAssist  # noqa: E402
from tool_transfer_bot.tasks.mdp.kinematic_joint_cmd import advance_bc_sketch_timestep  # noqa: E402

POLICY_JOINTS = ("joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "rh_r1")
DRAWER_OPEN_RAD = -0.2


def _resolve_checkpoint(agent_cfg) -> str | None:
    if args_cli.open_loop_demo:
        return None
    if args_cli.checkpoint:
        return retrieve_file_path(args_cli.checkpoint)
    log_root = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    return get_checkpoint_path(log_root, agent_cfg.load_run, agent_cfg.load_checkpoint)


def _joint_vec(robot) -> torch.Tensor:
    idx = [robot.joint_names.index(n) for n in POLICY_JOINTS]
    return robot.data.joint_pos[0, idx].clone()


def _load_demo_actions(demo_key: str, n_steps: int = 0) -> torch.Tensor:
    path = os.path.abspath(args_cli.dataset)
    with h5py.File(path, "r") as f:
        if demo_key not in f["data"]:
            raise KeyError(f"{demo_key} not in {path}")
        actions = f["data"][demo_key]["actions"]
        n = len(actions) if n_steps <= 0 else min(n_steps, len(actions))
        return torch.tensor(actions[:n], dtype=torch.float32)


def _step_env(
    env: RslRlVecEnvWrapper,
    actions: torch.Tensor,
    grasp_assist: GraspAssist | None = None,
):
    if args_cli.kinematic:
        advance_bc_sketch_timestep(
            env.unwrapped,
            actions[0, :7],
            interp_substeps=args_cli.interp_substeps,
        )
        if grasp_assist is not None:
            raw = env.unwrapped
            robot = raw.scene["robot"]
            raw.scene.update(dt=raw.physics_dt)
            grip = float(actions[0, 6].item())
            grasp_assist.update(robot, raw, grip)
            grasp_assist.reassert_tool(robot, raw)
        obs = env.get_observations()
        zeros = torch.zeros(env.num_envs, device=env.unwrapped.device)
        return obs, zeros, zeros.to(dtype=torch.long), {}
    return env.step(actions)


def main() -> None:
    device = args_cli.device if args_cli.device else "cuda:0"
    task_name = args_cli.task.split(":")[-1]
    env_cfg = parse_env_cfg(
        task_name,
        device=device,
        num_envs=args_cli.num_envs,
        use_fabric=False if args_cli.disable_fabric else None,
    )
    agent_cfg = load_cfg_from_registry(task_name, "rsl_rl_cfg_entry_point")
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))

    env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    raw = env.unwrapped
    robot = raw.scene["robot"]
    toolbox = raw.scene["toolbox"]
    drawer_idx = toolbox.joint_names.index("drawer_joint")

    policy = None
    ckpt = _resolve_checkpoint(agent_cfg)
    if ckpt is not None:
        print(f"[INFO] Loading checkpoint: {ckpt}")
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        runner.load(ckpt)
        policy = runner.get_inference_policy(device=raw.device)
    elif args_cli.open_loop_demo:
        print(f"[INFO] Open-loop replay from {args_cli.open_loop_demo}")

    compare_actions = None
    if args_cli.compare_demo:
        compare_actions = _load_demo_actions(args_cli.compare_demo).to(raw.device)

    grasp_assist: GraspAssist | None = None
    use_grasp_assist = (
        args_cli.kinematic
        and not args_cli.no_grasp_assist
        and (policy is not None or args_cli.open_loop_demo)
    )
    if use_grasp_assist:
        grasp_assist = GraspAssist()
        grasp_assist.capture_staging_pose(robot, raw)
        print("[INFO] grasp assist ON (table-lock + weld, matches demo collection)", flush=True)

    demo_actions = None
    if args_cli.open_loop_demo:
        demo_actions = _load_demo_actions(args_cli.open_loop_demo).to(raw.device)

    if args_cli.warmup_steps > 0 and not args_cli.open_loop_demo:
        warmup = _load_demo_actions(args_cli.warmup_demo, args_cli.warmup_steps).to(raw.device)
        print(
            f"[INFO] Warmup: {warmup.shape[0]} open-loop steps from {args_cli.warmup_demo} "
            f"(kinematic={args_cli.kinematic}, interp={args_cli.interp_substeps})",
            flush=True,
        )
        for t in range(warmup.shape[0]):
            _step_env(env, warmup[t].unsqueeze(0), grasp_assist)

    obs = env.get_observations()
    max_steps = args_cli.steps
    if demo_actions is not None:
        max_steps = min(max_steps, demo_actions.shape[0])

    print(
        f"[INFO] Running {max_steps} steps (log every {args_cli.log_interval}, "
        f"kinematic={args_cli.kinematic}, interp={args_cli.interp_substeps})",
        flush=True,
    )
    drawer0 = float(toolbox.data.joint_pos[0, drawer_idx].item())
    print(f"[INFO] drawer_joint={drawer0:.3f} (open target {DRAWER_OPEN_RAD})", flush=True)

    for step in range(max_steps):
        with torch.inference_mode():
            if demo_actions is not None:
                actions = demo_actions[step].unsqueeze(0)
            else:
                actions = policy(obs)
            obs, _, dones, _ = _step_env(env, actions, grasp_assist)
            if policy is not None and version.parse(metadata.version("rsl-rl-lib")) >= version.parse("4.0.0"):
                policy.reset(dones)

        if step % args_cli.log_interval == 0:
            q = _joint_vec(robot)
            a = actions[0].detach().cpu()
            err = (a - q.cpu()).abs()
            drawer = float(toolbox.data.joint_pos[0, drawer_idx].item())
            msg = (
                f"step {step:4d} | drawer={drawer:+.3f} | action j3={a[2]:+.3f} q3={q[2]:+.3f} "
                f"| max|a-q|={err.max():.3f} mean|a-q|={err.mean():.3f}"
            )
            if compare_actions is not None:
                demo_idx = args_cli.warmup_steps + step
                if demo_idx < compare_actions.shape[0]:
                    d = compare_actions[demo_idx].cpu()
                    demo_err = (a - d).abs()
                    msg += f" | demo j3={d[2]:+.3f} max|a-demo|={demo_err.max():.3f}"
            print(msg, flush=True)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
