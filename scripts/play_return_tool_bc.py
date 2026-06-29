"""Play ReturnTool BC checkpoints with kinematic joint replay (matches demo collection).

BC demos use sketch teleop (kinematic joint teleport), not PhysX PD tracking.

Usage:
    # BC policy (default: 32-step demo prefix warmup)
    ../IsaacLab/isaaclab.sh -p scripts/play_return_tool_bc.py \\
        --checkpoint logs/rsl_rl/return_tool_teacher/bc_warmstart/model_0.pt

    # HDF5 demo only — no policy (check whether motion is in the data itself)
    ../IsaacLab/isaaclab.sh -p scripts/play_return_tool_bc.py --open_loop_only

    # Smoother viewport (linear interp between joint targets)
    ../IsaacLab/isaaclab.sh -p scripts/play_return_tool_bc.py --checkpoint ... --interp_substeps 8
"""

from __future__ import annotations

import argparse
import os
import sys
import time

from isaaclab.app import AppLauncher

_ISAACLAB_RSL_RL = os.path.join(
    os.environ.get("ISAACLAB_PATH", "/home/user/IsaacLab"),
    "scripts",
    "reinforcement_learning",
    "rsl_rl",
)
sys.path.insert(0, _ISAACLAB_RSL_RL)
import cli_args  # noqa: E402

parser = argparse.ArgumentParser(description="ReturnTool BC play (kinematic joint replay).")
parser.add_argument("--task", type=str, default="Isaac-ReturnTool-Teacher-Play-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument(
    "--dataset",
    type=str,
    default="./data/demos/return_tool/dataset.hdf5",
    help="HDF5 for warmup / open-loop replay.",
)
parser.add_argument(
    "--warmup_demo",
    type=str,
    default="demo_0",
    help="Demo key for warmup or --open_loop_only.",
)
parser.add_argument(
    "--warmup_steps",
    type=int,
    default=32,
    help="Open-loop steps before policy (0=off). Demo pre-grasp hold ≈32.",
)
parser.add_argument(
    "--open_loop_only",
    action="store_true",
    help="Replay HDF5 actions only (no policy). Isolates demo trajectory vs network.",
)
parser.add_argument(
    "--open_loop_steps",
    type=int,
    default=0,
    help="Open-loop length when --open_loop_only (0 = full demo).",
)
parser.add_argument(
    "--interp_substeps",
    type=int,
    default=1,
    help="Linear joint interp per control step (1=teleport like demo record, 8=smoother).",
)
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O."
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
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
from tool_transfer_bot.tasks.mdp.kinematic_joint_cmd import advance_bc_sketch_timestep  # noqa: E402


def _resolve_checkpoint(agent_cfg) -> str | None:
    if args_cli.open_loop_only:
        return None
    if args_cli.checkpoint:
        return retrieve_file_path(args_cli.checkpoint)
    log_root = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    return get_checkpoint_path(log_root, args_cli.load_run, agent_cfg.load_checkpoint)


def _load_demo_actions(demo_key: str, n_steps: int) -> torch.Tensor:
    path = os.path.abspath(args_cli.dataset)
    with h5py.File(path, "r") as f:
        if demo_key not in f["data"]:
            raise KeyError(f"{demo_key} not in {path}")
        actions = f["data"][demo_key]["actions"]
        n = len(actions) if n_steps <= 0 else min(n_steps, len(actions))
        return torch.tensor(actions[:n], dtype=torch.float32)


def _step_bc_sketch(env: RslRlVecEnvWrapper, actions: torch.Tensor):
    advance_bc_sketch_timestep(
        env.unwrapped,
        actions[0, :7],
        interp_substeps=args_cli.interp_substeps,
    )
    obs = env.get_observations()
    zeros = torch.zeros(env.num_envs, device=env.unwrapped.device)
    return obs, zeros, zeros.to(dtype=torch.long), {}


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
    toolbox = raw.scene["toolbox"]
    drawer_idx = toolbox.joint_names.index("drawer_joint")
    print(
        f"[INFO] drawer_joint={toolbox.data.joint_pos[0, drawer_idx].item():.3f} (open ≈ -0.2)",
        flush=True,
    )

    policy = None
    ckpt = _resolve_checkpoint(agent_cfg)
    if ckpt is not None:
        print(f"[INFO] Loading checkpoint: {ckpt}")
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        runner.load(ckpt)
        policy = runner.get_inference_policy(device=raw.device)

    if args_cli.open_loop_only:
        loop_actions = _load_demo_actions(args_cli.warmup_demo, args_cli.open_loop_steps).to(raw.device)
        print(
            f"[INFO] Open-loop only: {loop_actions.shape[0]} steps from {args_cli.warmup_demo} "
            f"(interp_substeps={args_cli.interp_substeps})",
            flush=True,
        )
        for t in range(loop_actions.shape[0]):
            if not simulation_app.is_running():
                break
            _step_bc_sketch(env, loop_actions[t].unsqueeze(0))
        print("[INFO] Demo replay finished. Close window to exit.", flush=True)
        while simulation_app.is_running():
            time.sleep(0.05)
        env.close()
        return

    warmup = None
    if args_cli.warmup_steps > 0:
        warmup = _load_demo_actions(args_cli.warmup_demo, args_cli.warmup_steps).to(raw.device)
        print(
            f"[INFO] Warmup: {warmup.shape[0]} open-loop steps from {args_cli.warmup_demo} "
            f"(steps 32+ in demo include large j2 elbow swing — see docs)",
            flush=True,
        )
        for t in range(warmup.shape[0]):
            _step_bc_sketch(env, warmup[t].unsqueeze(0))

    obs = env.get_observations()
    dt = raw.step_dt
    print(
        f"[INFO] Policy loop (interp_substeps={args_cli.interp_substeps}). Close window to exit.",
        flush=True,
    )

    while simulation_app.is_running():
        start = time.time()
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, dones, _ = _step_bc_sketch(env, actions)
            if version.parse(metadata.version("rsl-rl-lib")) >= version.parse("4.0.0"):
                policy.reset(dones)

        if args_cli.real_time:
            sleep_time = dt - (time.time() - start)
            if sleep_time > 0:
                time.sleep(sleep_time)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
