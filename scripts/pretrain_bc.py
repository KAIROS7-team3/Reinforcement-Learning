#!/usr/bin/env python3
"""Offline BC pretrain from HDF5 demos → rsl_rl checkpoint for PPO warm-start.

Usage:
    cd /home/user/Reinforcement-Learning
    ../IsaacLab/isaaclab.sh -p scripts/pretrain_bc.py \\
        --task Isaac-ReturnTool-Teacher-v0 \\
        --dataset ./data/demos/return_tool/dataset.hdf5 \\
        --output ./data/checkpoints/return_tool/bc_warmstart.pt \\
        --epochs 200
"""

from __future__ import annotations

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="BC pretrain from HDF5 demos.")
parser.add_argument("--task", type=str, default="Isaac-ReturnTool-Teacher-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--dataset", type=str, required=True)
parser.add_argument("--output", type=str, default="./data/checkpoints/return_tool/bc_warmstart.pt")
parser.add_argument("--epochs", type=int, default=200)
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--val_ratio", type=float, default=0.1)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument(
    "--exclude_demos",
    type=str,
    default="",
    help="Comma-separated episode keys to skip (e.g. demo_21,demo_35).",
)
parser.add_argument(
    "--obs_noise_std",
    type=float,
    default=0.0,
    help="Gaussian noise on train obs (joint dims, rad). 0=off. Try 0.02 for closed-loop robustness.",
)
parser.add_argument(
    "--object_noise_std",
    type=float,
    default=0.0,
    help="Gaussian noise on train obs object_pos (dims 7:10, m). Try 0.005 with --obs_noise_std.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "source"))
import tool_transfer_bot  # noqa: F401, E402

import importlib.metadata as metadata  # noqa: E402
import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from packaging import version  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402
from tensordict import TensorDict  # noqa: E402

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry, parse_env_cfg  # noqa: E402

from tool_transfer_bot.agents.demo_buffer import DemoTransitionDataset  # noqa: E402

POLICY_OBS_GROUP = "policy"
JOINT_OBS_SLICE = slice(0, 7)
OBJECT_OBS_SLICE = slice(7, 10)


def _augment_obs(obs: torch.Tensor) -> torch.Tensor:
    """Add train-time noise on joint + object channels (closed-loop robustness)."""
    if args_cli.obs_noise_std <= 0.0 and args_cli.object_noise_std <= 0.0:
        return obs
    noisy = obs.clone()
    if args_cli.obs_noise_std > 0.0:
        noisy[..., JOINT_OBS_SLICE] += torch.randn_like(noisy[..., JOINT_OBS_SLICE]) * args_cli.obs_noise_std
    if args_cli.object_noise_std > 0.0:
        noisy[..., OBJECT_OBS_SLICE] += (
            torch.randn_like(noisy[..., OBJECT_OBS_SLICE]) * args_cli.object_noise_std
        )
    return noisy


def _parse_exclude_demos() -> frozenset[str]:
    raw = args_cli.exclude_demos.strip()
    if not raw:
        return frozenset()
    return frozenset(x.strip() for x in raw.split(",") if x.strip())


def _resolve_actor(runner: OnPolicyRunner) -> nn.Module:
    """Return the trainable policy network across rsl-rl API versions."""
    alg = runner.alg
    if hasattr(alg, "actor"):
        return alg.actor
    if hasattr(alg, "policy"):
        return alg.policy
    if hasattr(alg, "actor_critic"):
        return alg.actor_critic.actor
    raise AttributeError(f"Unsupported rsl-rl algorithm type: {type(alg)}")


def _obs_batch(obs: torch.Tensor, device: torch.device) -> TensorDict | torch.Tensor:
    """Build rsl-rl >= 4.0 TensorDict obs; legacy versions use flat tensors."""
    obs = obs.to(device)
    rsl_version = version.parse(metadata.version("rsl-rl-lib"))
    if rsl_version >= version.parse("4.0.0"):
        return TensorDict({POLICY_OBS_GROUP: obs}, batch_size=[obs.shape[0]], device=device)
    return obs


def _predict_actions(actor: nn.Module, obs: TensorDict | torch.Tensor) -> torch.Tensor:
    if isinstance(obs, TensorDict):
        return actor(obs)
    return actor(obs)


def _set_train_mode(runner: OnPolicyRunner, train: bool) -> None:
    alg = runner.alg
    if hasattr(alg, "train_mode") and hasattr(alg, "eval_mode"):
        alg.train_mode() if train else alg.eval_mode()
    else:
        actor = _resolve_actor(runner)
        actor.train(train)


def main() -> None:
    torch.manual_seed(args_cli.seed)
    device = torch.device(args_cli.device if args_cli.device else "cuda:0")

    task_name = args_cli.task.split(":")[-1]
    env_cfg = parse_env_cfg(task_name, device=str(device), num_envs=args_cli.num_envs)
    agent_cfg = load_cfg_from_registry(task_name, "rsl_rl_cfg_entry_point")
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))

    env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir="/tmp/bc_pretrain", device=agent_cfg.device)
    actor = _resolve_actor(runner)

    dataset = DemoTransitionDataset(
        args_cli.dataset,
        device="cpu",
        exclude_episodes=_parse_exclude_demos(),
    )
    policy_obs_dim = int(env.unwrapped.observation_manager.group_obs_dim["policy"][0])
    env_action_dim = int(env.unwrapped.action_manager.total_action_dim)
    if dataset.obs_dim != policy_obs_dim:
        raise ValueError(
            f"Demo obs_dim={dataset.obs_dim} != Teacher policy obs_dim={policy_obs_dim}. "
            "Re-collect demos with the same observation terms as Isaac-ReturnTool-Teacher-v0."
        )
    if dataset.action_dim != env_action_dim:
        raise ValueError(
            f"Demo action_dim={dataset.action_dim} != Teacher action_dim={env_action_dim}. "
            "Use Isaac-ReturnTool-Teacher-Demo-v0 (7D joint targets) for collection."
        )

    n_val = max(1, int(len(dataset) * args_cli.val_ratio))
    n_train = len(dataset) - n_val
    train_ds, val_ds = torch.utils.data.random_split(
        dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(args_cli.seed),
    )
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args_cli.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=args_cli.batch_size, shuffle=False)

    print(
        f"[INFO] transitions={len(dataset)} obs_dim={dataset.obs_dim} action_dim={dataset.action_dim} "
        f"(Teacher policy={policy_obs_dim}, action={env_action_dim})"
    )
    print(f"[INFO] train={n_train} val={n_val} epochs={args_cli.epochs}")
    if args_cli.exclude_demos:
        print(f"[INFO] excluded demos: {sorted(_parse_exclude_demos())}")
    if args_cli.obs_noise_std > 0.0 or args_cli.object_noise_std > 0.0:
        print(
            f"[INFO] train obs noise: joint_std={args_cli.obs_noise_std} "
            f"object_std={args_cli.object_noise_std}",
        )
    print("[INFO] BC uses supervised MSE on demo actions (no env reward).")

    optimizer = torch.optim.Adam(actor.parameters(), lr=args_cli.lr)
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    best_actor_state: dict[str, torch.Tensor] | None = None

    for epoch in range(args_cli.epochs):
        _set_train_mode(runner, train=True)
        train_loss = 0.0
        for obs, actions in train_loader:
            obs = _obs_batch(_augment_obs(obs), device)
            actions = actions.to(device)
            if isinstance(obs, TensorDict) and hasattr(actor, "update_normalization"):
                actor.update_normalization(obs)
            pred = _predict_actions(actor, obs)
            loss = loss_fn(pred, actions)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= max(1, len(train_loader))

        _set_train_mode(runner, train=False)
        val_loss = 0.0
        with torch.no_grad():
            for obs, actions in val_loader:
                obs = _obs_batch(obs, device)
                actions = actions.to(device)
                pred = _predict_actions(actor, obs)
                val_loss += loss_fn(pred, actions).item()
        val_loss /= max(1, len(val_loader))

        if val_loss < best_val:
            best_val = val_loss
            best_actor_state = {k: v.cpu().clone() for k, v in actor.state_dict().items()}

        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"[INFO] epoch {epoch + 1}/{args_cli.epochs} train_mse={train_loss:.6f} val_mse={val_loss:.6f}")

    if best_actor_state is not None:
        actor.load_state_dict(best_actor_state)

    out_path = os.path.abspath(args_cli.output)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    # runner.save() uploads via logger (needs tensorboard/wandb writer); save alg state directly.
    saved_dict = runner.alg.save()
    saved_dict["iter"] = 0
    saved_dict["infos"] = {"bc_pretrain": True, "best_val_mse": best_val}
    torch.save(saved_dict, out_path)
    print(f"[INFO] Saved BC checkpoint → {out_path} (best val_mse={best_val:.6f})")
    print(
        "[INFO] PPO warm-start:\n"
        f"  mkdir -p logs/rsl_rl/{agent_cfg.experiment_name}/bc_warmstart\n"
        f"  cp {out_path} logs/rsl_rl/{agent_cfg.experiment_name}/bc_warmstart/model_0.pt\n"
        f"  isaaclab.sh -p scripts/train.py --task {args_cli.task} --headless --resume "
        f"--load_run bc_warmstart --checkpoint model_0.pt --run_name bc_ppo_v1"
    )

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
