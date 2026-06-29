"""Compare drawer visual vs marker (FrameTransformer) with Fabric ON vs OFF.

Measures after sim settle:
  - Knob mesh world bbox (visual)
  - drawer_frame source (PhysX body) and target (handle marker)
  - Articulation drawer link pose

Usage:
    ./isaaclab.sh -p scripts/compare_drawer_fabric.py --headless
"""

from __future__ import annotations

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Compare drawer alignment Fabric ON vs OFF.")
parser.add_argument("--task", type=str, default="Isaac-OpenDrawer-Teacher-Play-v0")
parser.add_argument("--num_envs", type=int, default=1)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "source"))
import tool_transfer_bot  # noqa: F401, E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from pxr import Gf, Usd, UsdGeom  # noqa: E402

from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

KNOB_SUFFIX = "/toolbox/toolbox/drawer/FurnitureKnob_01/Mesh"
DRAWER_SUFFIX = "/toolbox/toolbox/drawer"
SETTLE_STEPS = 24


def _knob_center_world(env) -> tuple[float, float, float]:
    stage = env.unwrapped.sim.stage
    cache = UsdGeom.XformCache()
    cache.SetTime(Usd.TimeCode.Default())
    for prim in stage.Traverse():
        path = prim.GetPath().pathString
        if path.endswith(KNOB_SUFFIX):
            xf = cache.GetLocalToWorldTransform(prim)
            pts = UsdGeom.Mesh(prim).GetPointsAttr().Get()
            xs, ys, zs = [], [], []
            for p in pts:
                w = xf.Transform(Gf.Vec3d(p[0], p[1], p[2]))
                xs.append(w[0])
                ys.append(w[1])
                zs.append(w[2])
            return (
                (min(xs) + max(xs)) * 0.5,
                (min(ys) + max(ys)) * 0.5,
                (min(zs) + max(zs)) * 0.5,
            )
    raise RuntimeError(f"Knob mesh not found (suffix {KNOB_SUFFIX})")


def _drawer_link_world(env) -> tuple[float, float, float]:
    stage = env.unwrapped.sim.stage
    cache = UsdGeom.XformCache()
    cache.SetTime(Usd.TimeCode.Default())
    for prim in stage.Traverse():
        path = prim.GetPath().pathString
        if path.endswith(DRAWER_SUFFIX) and prim.GetName() == "drawer":
            t = cache.GetLocalToWorldTransform(prim).ExtractTranslation()
            return (t[0], t[1], t[2])
    raise RuntimeError(f"Drawer link not found (suffix {DRAWER_SUFFIX})")


def _measure(env, label: str) -> dict:
    for _ in range(SETTLE_STEPS):
        env.unwrapped.sim.step()

    scene = env.unwrapped.scene
    drawer_frame = scene["drawer_frame"]
    toolbox = scene["toolbox"]

    source = drawer_frame.data.source_pos_w[0].cpu()
    target = drawer_frame.data.target_pos_w[0, 0].cpu()
    knob = torch.tensor(_knob_center_world(env), device=source.device)
    link = torch.tensor(_drawer_link_world(env), device=source.device)

    drawer_idx = toolbox.find_bodies("drawer")[0][0]
    link_art = toolbox.data.body_link_pos_w[0, drawer_idx].cpu()

    return {
        "label": label,
        "use_fabric": env.unwrapped.sim.cfg.use_fabric,
        "knob_visual": knob,
        "drawer_link_usd": link,
        "body_link_art": link_art,
        "marker_source": source,
        "marker_target": target,
        "target_minus_knob": target - knob,
        "source_minus_knob": source - knob,
    }


def _print_row(name: str, on: dict, off: dict, key: str) -> None:
    a = on[key]
    b = off[key]
    if isinstance(a, torch.Tensor):
        da = (a - b).abs().max().item()
        print(f"  {name:22}  ON {tuple(round(x, 5) for x in a.tolist())}")
        print(f"  {'':22}  OFF {tuple(round(x, 5) for x in b.tolist())}")
        print(f"  {'':22}  |Δ|_max = {da:.6f} m")
    else:
        print(f"  {name:22}  ON={a}  OFF={b}")


def main() -> None:
    results: list[dict] = []
    for use_fabric, label in [(True, "Fabric ON"), (False, "Fabric OFF")]:
        env_cfg = parse_env_cfg(
            args_cli.task,
            device=args_cli.device,
            num_envs=args_cli.num_envs,
            use_fabric=use_fabric,
        )
        env_cfg.scene.drawer_frame.debug_vis = False
        print(f"\n=== {label} (use_fabric={use_fabric}) ===", flush=True)
        env = gym.make(args_cli.task, cfg=env_cfg)
        env.reset()
        results.append(_measure(env, label))
        env.close()

    on, off = results[0], results[1]
    print("\n" + "=" * 60)
    print("FABRIC ON vs OFF comparison (closed drawer)")
    print("=" * 60)
    _print_row("knob visual (world)", on, off, "knob_visual")
    _print_row("drawer link USD", on, off, "drawer_link_usd")
    _print_row("drawer link articulation", on, off, "body_link_art")
    _print_row("marker source (body)", on, off, "marker_source")
    _print_row("marker target (handle)", on, off, "marker_target")
    _print_row("target − knob", on, off, "target_minus_knob")
    _print_row("source − knob", on, off, "source_minus_knob")

    tgt_err_on = on["target_minus_knob"].norm().item()
    tgt_err_off = off["target_minus_knob"].norm().item()
    print()
    print(f"  Handle marker error (‖target−knob‖):  ON={tgt_err_on:.6f} m  OFF={tgt_err_off:.6f} m")
    vis_delta = (on["knob_visual"] - off["knob_visual"]).abs().max().item()
    print(f"  Knob visual shift (|ON−OFF|_max):     {vis_delta:.6f} m")
    if tgt_err_off < 0.002 and vis_delta > 0.005:
        print("  → Marker OK on Fabric OFF; visual mesh differs on Fabric ON.")
    elif tgt_err_on < 0.002 and tgt_err_off < 0.002:
        print("  → Marker and visual agree on both paths.")
    elif vis_delta < 0.002 and tgt_err_on > 0.005:
        print("  → Visual same; marker/body path differs with Fabric ON.")


if __name__ == "__main__":
    main()
    simulation_app.close()
