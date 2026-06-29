"""Reset OpenDrawer env, print toolbox state, save viewer screenshot."""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Isaac-OpenDrawer-Teacher-Play-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument(
    "--out",
    type=str,
    default="/home/user/Reinforcement-Learning/logs/toolbox_verify.png",
)
parser.add_argument(
    "--keep_open",
    action="store_true",
    default=False,
    help="Keep GUI window open until closed manually (no --headless).",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "source"))
import tool_transfer_bot  # noqa: F401, E402

import gymnasium as gym  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402


def _save_rgb(path: str, rgb: np.ndarray) -> None:
    from PIL import Image

    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(rgb[:, :, :3]).save(path)


def _capture_via_env_render(env) -> np.ndarray | None:
    for _ in range(24):
        env.unwrapped.sim.render()
        simulation_app.update()
        rgb = env.render()
        if rgb is not None and rgb.size > 0 and rgb.max() > 0:
            return rgb
    return rgb if rgb is not None and rgb.size > 0 else None


def _capture_via_viewport(path: str) -> bool:
    try:
        import omni.kit.viewport.utility as vp_util

        vp = vp_util.get_active_viewport()
        if vp is None:
            return False
        vp.viewport_api.capture_image(path)
        return os.path.isfile(path)
    except Exception:
        return False


def main() -> None:
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    render_mode = "rgb_array" if args_cli.headless else None
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=render_mode)
    env.reset()

    tb = env.unwrapped.scene["toolbox"]
    handle = env.unwrapped.scene["drawer_frame"].data.target_pos_w[0, 0]
    print("=== toolbox verify ===", flush=True)
    print(f"joint_names: {list(tb.joint_names)}", flush=True)
    print(f"body_names:  {list(tb.body_names)}", flush=True)
    print(f"drawer_joint_pos: {tb.data.joint_pos[0].cpu().numpy()}", flush=True)
    print(f"handle_pos_w:     {handle.cpu().numpy()}", flush=True)

    saved = False
    if args_cli.headless:
        rgb = _capture_via_env_render(env)
        if rgb is not None:
            _save_rgb(args_cli.out, rgb)
            saved = True
            print(f"screenshot: {args_cli.out}  shape={rgb.shape}", flush=True)
        else:
            print("screenshot: failed (headless render returned empty)", flush=True)
    else:
        for _ in range(8):
            env.unwrapped.sim.step()
            simulation_app.update()
        if _capture_via_viewport(args_cli.out):
            saved = True
            print(f"screenshot: {args_cli.out}", flush=True)
        else:
            print("screenshot: skipped (no active viewport)", flush=True)

    if args_cli.keep_open and not args_cli.headless:
        # Hold the reset pose — do not call env.step(zero): DiffIK re-solves every policy
        # step and stiff implicit actuators amplify numerical jitter into visible wobble.
        print("[INFO] GUI open (physics only) — close the Isaac Sim window to exit.", flush=True)
        while simulation_app.is_running():
            env.unwrapped.sim.step()
            simulation_app.update()
    else:
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
