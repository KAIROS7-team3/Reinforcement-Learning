"""Dump knob-frame and drawer poses after reset (headless sanity check).

Prints ``drawer_frame`` (reward aim = FurnitureKnob_01 / drawer_handle_top), not prim
``handle`` (toolbox top carry handle).
"""

from __future__ import annotations

import argparse
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
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402


def main() -> None:
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env = gym.make(args_cli.task, cfg=env_cfg)
    env.reset()
    scene = env.unwrapped.scene
    knob_frame = scene["drawer_frame"].data.target_pos_w[0, 0]
    tb = scene["toolbox"]
    joint_names = list(tb.joint_names)
    lines = [
        f"drawer_frame (knob aim): {knob_frame.cpu().numpy()}",
        f"root_pos_w: {tb.data.root_pos_w[0].cpu().numpy()}",
        f"joint_names: {joint_names}",
        f"drawer_joint_pos: {tb.data.joint_pos[0, tb.find_joints('drawer_joint')[0]].item():.6f}",
    ]
    for i, name in enumerate(tb.body_names):
        lines.append(f"body[{name}]: {tb.data.body_pos_w[0, i].cpu().numpy()}")

    import omni.usd
    from pxr import Gf, UsdGeom

    stage = omni.usd.get_context().get_stage()
    cache = UsdGeom.XformCache()
    for suffix in [
        "toolbox/toolbox/toolbox",
        "toolbox/toolbox/toolbox/drawer/drawer_handle_top",
        "toolbox/toolbox/toolbox/drawer/FurnitureKnob_01/Mesh",
        "toolbox/toolbox/toolbox/handle/handle",  # toolbox top carry handle (NOT knob)
        "toolbox/toolbox/static_drawer_02",
    ]:
        path = f"/World/envs/env_0/toolbox_with_handle/{suffix}"
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid():
            continue
        t = cache.GetLocalToWorldTransform(prim).ExtractTranslation()
        if prim.IsA(UsdGeom.Mesh):
            pts = UsdGeom.Mesh(prim).GetPointsAttr().Get()
            m = cache.GetLocalToWorldTransform(prim)
            ws = [m.Transform(Gf.Vec3d(p[0], p[1], p[2])) for p in pts]
            xs = [w[0] for w in ws]
            ys = [w[1] for w in ws]
            zs = [w[2] for w in ws]
            cx, cy, cz = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2, (min(zs) + max(zs)) / 2
            lines.append(f"{suffix.split('/')[-1]} xform=({t[0]:.4f},{t[1]:.4f},{t[2]:.4f}) bbox=({cx:.4f},{cy:.4f},{cz:.4f})")
        else:
            lines.append(f"{suffix} xform=({t[0]:.4f},{t[1]:.4f},{t[2]:.4f})")

    text = "\n".join(lines)
    print(text, flush=True)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
