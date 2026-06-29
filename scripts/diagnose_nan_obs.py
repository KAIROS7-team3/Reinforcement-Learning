"""Find which policy observation term first becomes NaN/Inf.

Usage (from Reinforcement-Learning, inside isaaclab.sh):
    cd ~/Reinforcement-Learning
    ../IsaacLab/isaaclab.sh -p scripts/diagnose_nan_obs.py --headless \\
        --task Isaac-OpenDrawer-Teacher-v0 --num_envs 64 --steps 5000

With the crashed checkpoint:
    ../IsaacLab/isaaclab.sh -p scripts/diagnose_nan_obs.py --headless \\
        --task Isaac-OpenDrawer-Teacher-v0 --num_envs 64 --steps 20000 \\
        --checkpoint logs/rsl_rl/open_drawer_teacher/2026-06-16_00-58-27_drawer_fabric_on_v1/model_1100.pt

Modes:
    policy  — actions from loaded checkpoint (default if --checkpoint set)
    random  — Uniform[-1, 1] actions
    zero    — zero actions
"""

from __future__ import annotations

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Diagnose NaN/Inf in policy observations.")
parser.add_argument("--task", type=str, default="Isaac-OpenDrawer-Teacher-v0")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--steps", type=int, default=5000)
parser.add_argument(
    "--action_mode",
    type=str,
    default="auto",
    choices=("auto", "policy", "random", "zero"),
    help="auto = policy if --checkpoint else random",
)
parser.add_argument("--checkpoint", type=str, default=None)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric (default: keep task cfg)."
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent  # noqa: E402
from isaaclab.utils.assets import retrieve_file_path  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry, parse_env_cfg  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "source"))
import tool_transfer_bot  # noqa: F401, E402


def _as_tensor(x) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x
    if hasattr(x, "get"):
        return x.get("policy")
    if isinstance(x, dict):
        return x["policy"]
    raise TypeError(f"Expected Tensor or TensorDict, got {type(x)}")


def _policy_obs_tensor(obs) -> torch.Tensor:
    if hasattr(obs, "get"):
        return obs.get("policy")
    if isinstance(obs, dict):
        return obs["policy"]
    return obs


def _bad_mask(t: torch.Tensor) -> torch.Tensor:
    return torch.isnan(t) | torch.isinf(t)


def _first_bad_envs(t: torch.Tensor, max_show: int = 8) -> list[int]:
    flat = _bad_mask(t).view(t.shape[0], -1).any(dim=1)
    ids = flat.nonzero(as_tuple=False).view(-1).tolist()
    return ids[:max_show]


def _check_tensor(name: str, t: torch.Tensor) -> tuple[bool, list[int]]:
    if t is None:
        return False, []
    bad = _bad_mask(t)
    if not bad.any():
        return False, []
    return True, _first_bad_envs(t)


def _scan_terms(obs_mgr, group: str) -> dict[str, torch.Tensor]:
    """Evaluate each observation term separately (group may concatenate)."""
    from isaaclab.utils import modifiers, noise

    env = obs_mgr._env
    term_names = obs_mgr.active_terms[group]
    term_cfgs = obs_mgr._group_obs_term_cfgs[group]
    out: dict[str, torch.Tensor] = {}
    for term_name, term_cfg in zip(term_names, term_cfgs):
        obs: torch.Tensor = term_cfg.func(env, **term_cfg.params).clone()
        if term_cfg.modifiers is not None:
            for modifier in term_cfg.modifiers:
                obs = modifier.func(obs, **modifier.params)
        if isinstance(term_cfg.noise, noise.NoiseCfg):
            obs = term_cfg.noise.func(obs, term_cfg.noise)
        elif isinstance(term_cfg.noise, noise.NoiseModelCfg) and term_cfg.noise.func is not None:
            obs = term_cfg.noise.func(obs)
        if term_cfg.clip:
            obs = obs.clip_(min=term_cfg.clip[0], max=term_cfg.clip[1])
        if term_cfg.scale is not None:
            obs = obs.mul_(term_cfg.scale)
        out[term_name] = obs
    return out


def _scan_scene(env) -> list[tuple[str, torch.Tensor]]:
    robot = env.scene["robot"]
    toolbox = env.scene["toolbox"]
    ee = env.scene["ee_frame"].data
    handle = env.scene["drawer_frame"].data
    rows: list[tuple[str, torch.Tensor]] = [
        ("robot.joint_pos", robot.data.joint_pos),
        ("robot.joint_vel", robot.data.joint_vel),
        ("robot.root_pos_w", robot.data.root_pos_w),
        ("robot.root_quat_w", robot.data.root_quat_w),
        ("toolbox.joint_pos", toolbox.data.joint_pos),
        ("toolbox.joint_vel", toolbox.data.joint_vel),
        ("ee_frame.target_pos_w", ee.target_pos_w),
        ("ee_frame.target_quat_w", ee.target_quat_w),
        ("drawer_frame.target_pos_w", handle.target_pos_w),
        ("drawer_frame.target_quat_w", handle.target_quat_w),
    ]
    if hasattr(env, "action_manager"):
        try:
            rows.append(("last_action", env.action_manager.action))
        except Exception:
            pass
    return rows


def _load_policy(env, task_name: str, checkpoint: str):
    import importlib.metadata as metadata
    from packaging import version
    from rsl_rl.runners import OnPolicyRunner

    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg

    _ISAACLAB_RSL_RL = os.path.join(
        os.environ.get("ISAACLAB_PATH", "/home/user/IsaacLab"),
        "scripts",
        "reinforcement_learning",
        "rsl_rl",
    )
    sys.path.insert(0, _ISAACLAB_RSL_RL)
    import cli_args  # noqa: E402

    agent_cfg = load_cfg_from_registry(task_name, "rsl_rl_cfg_entry_point")
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))
    agent_cfg.device = args_cli.device

    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(wrapped, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(retrieve_file_path(checkpoint))
    policy = runner.get_inference_policy(device=wrapped.device)
    return wrapped, policy


def main() -> None:
    task_name = args_cli.task.split(":")[-1]
    use_fabric = False if args_cli.disable_fabric else None
    env_cfg = parse_env_cfg(
        task_name,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=use_fabric,
    )

    torch.manual_seed(args_cli.seed)
    env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    action_mode = args_cli.action_mode
    if action_mode == "auto":
        action_mode = "policy" if args_cli.checkpoint else "random"

    policy = None
    vec_env = env
    if action_mode == "policy":
        if not args_cli.checkpoint:
            raise ValueError("--checkpoint required for action_mode=policy")
        vec_env, policy = _load_policy(env, task_name, args_cli.checkpoint)
        print(f"[INFO] Loaded checkpoint: {args_cli.checkpoint}")
    else:
        print(f"[INFO] Action mode: {action_mode}")

    unwrapped = vec_env.unwrapped
    obs_mgr = unwrapped.observation_manager
    print(f"[INFO] Policy obs terms: {obs_mgr.active_terms.get('policy', [])}")

    obs, _ = vec_env.reset()
    print(f"[INFO] Running {args_cli.steps} steps x {args_cli.num_envs} envs")

    first_hit: dict | None = None

    for step in range(args_cli.steps):
        if action_mode == "policy":
            with torch.inference_mode():
                raw_actions = policy(obs)
                actions = raw_actions if isinstance(raw_actions, torch.Tensor) else _as_tensor(raw_actions)
        elif action_mode == "zero":
            actions = torch.zeros(vec_env.action_space.shape, device=vec_env.unwrapped.device)
        else:
            actions = torch.empty(vec_env.action_space.shape, device=vec_env.unwrapped.device).uniform_(-1.0, 1.0)

        if _bad_mask(actions).any():
            bad_envs = _first_bad_envs(actions)
            first_hit = {
                "step": step,
                "where": "actions (before step)",
                "term": "policy_output",
                "env_ids": bad_envs,
            }
            break

        obs, rewards, dones, _ = vec_env.step(actions)

        if _bad_mask(rewards).any():
            first_hit = {
                "step": step,
                "where": "rewards",
                "term": "total_reward",
                "env_ids": _first_bad_envs(rewards.unsqueeze(-1)),
            }
            break

        for group in ("policy",):
            terms = _scan_terms(obs_mgr, group)
            for term_name, tensor in terms.items():
                hit, env_ids = _check_tensor(term_name, tensor)
                if hit:
                    first_hit = {
                        "step": step,
                        "where": f"obs/{group}",
                        "term": term_name,
                        "env_ids": env_ids,
                        "sample": tensor[env_ids[0]].detach().cpu().tolist() if env_ids else None,
                    }
                    break
            if first_hit:
                break

        if first_hit:
            break

        policy_obs = _policy_obs_tensor(obs)
        if _bad_mask(policy_obs).any():
            first_hit = {
                "step": step,
                "where": "obs/policy (concatenated only; check per-term above)",
                "term": "policy",
                "env_ids": _first_bad_envs(policy_obs),
            }
            break

        for name, tensor in _scan_scene(unwrapped):
            hit, env_ids = _check_tensor(name, tensor)
            if hit:
                first_hit = {
                    "step": step,
                    "where": "scene",
                    "term": name,
                    "env_ids": env_ids,
                    "sample": tensor[env_ids[0]].detach().cpu().tolist() if env_ids else None,
                }
                break
        if first_hit:
            break

        if step > 0 and step % 1000 == 0:
            policy_obs = _policy_obs_tensor(obs)
            print(
                f"  step {step}: ok | reward mean {rewards.mean().item():.3f} | "
                f"policy obs [{policy_obs.min().item():.3f}, {policy_obs.max().item():.3f}]"
            )

    if first_hit:
        print("\n=== FIRST NaN/Inf DETECTED ===")
        for k, v in first_hit.items():
            print(f"  {k}: {v}")
        print("\nLikely root cause is upstream of this term (physics, IK, or frame pose).")
    else:
        print(f"\n=== No NaN/Inf in {args_cli.steps} steps ===")
        print("Try more steps, --num_envs 4096, or the same checkpoint with --steps 50000.")

    vec_env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
