#!/usr/bin/env python3
"""Verify toolbox drawer layout for with_camera.usda before RL training.

Checks rebaked desk USD + with_camera payload wiring. Articulated drawers
only align in viewport after Play (PhysX resolves drawer_joint); this script
reports both the raw USD xform view and the physics-closed pose.
"""

from __future__ import annotations

import argparse
import os
import sys

from pxr import Gf, Usd, UsdGeom, UsdPhysics

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)

from fix_toolbox_xform import (  # noqa: E402
    ROOT,
    ROOT_LINK,
    STATIC_BOTTOM,
    TOP_DRAWER_LINK,
    TOP_JOINT,
    _closed_top_drawer_z_in_root,
    _mesh_z_range_in_root,
    _validate_desk,
)

DEFAULT_SCENE = "/home/user/Desktop/with_camera.usda"
DEFAULT_DESK = "/home/user/Desktop/toolbox_rl_desk.usda"
DEFAULT_FLAT = os.path.join(REPO_ROOT, "assets", "toolbox_rl_flat.usda")


def _world_path(scene_prefix: str, desk_path: str) -> str:
    """Map desk-root path to with_camera world path."""
    suffix = desk_path.removeprefix("/toolbox_with_handle")
    return f"{scene_prefix}{suffix}"


def _check_payload(scene_path: str, desk_path: str) -> None:
    stage = Usd.Stage.Open(scene_path)
    prim = stage.GetPrimAtPath("/World/toolbox_with_handle")
    if not prim.IsValid():
        raise RuntimeError("Missing /World/toolbox_with_handle in with_camera.usda")
    payloads = prim.GetPrimStack()
    payload_files = []
    for spec in payloads:
        if spec.payloadList:
            for p in spec.payloadList.prependedItems:
                payload_files.append(p.assetPath)
            for p in spec.payloadList.appendedItems:
                payload_files.append(p.assetPath)
    desk_abs = os.path.abspath(desk_path)
    scene_dir = os.path.dirname(os.path.abspath(scene_path))
    expected = os.path.abspath(os.path.join(scene_dir, "toolbox_rl_desk.usda"))
    print(f"Scene: {scene_path}")
    print(f"Desk payload expected: {expected}")
    print(f"Desk file exists: {os.path.isfile(desk_abs)}")
    if not os.path.isfile(desk_abs):
        raise FileNotFoundError(f"Missing desk USD: {desk_abs}")
    if expected != desk_abs:
        print("  WARN: with_camera payload path may not match --desk path")


def _report_with_camera(scene_path: str, desk_path: str) -> None:
    scene = Usd.Stage.Open(scene_path)
    desk = Usd.Stage.Open(desk_path)
    if scene is None or desk is None:
        raise RuntimeError("Failed to open scene or desk USD")

    prefix = "/World/toolbox_with_handle"
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())

    joint_path = _world_path(prefix, TOP_JOINT)
    joint = scene.GetPrimAtPath(joint_path)
    if not joint.IsValid():
        raise RuntimeError(f"Missing joint: {joint_path}")

    pos = joint.GetAttribute("state:linear:physics:position").Get()
    print(f"\n[with_camera] drawer_joint position: {pos} (0 = closed)")

    # Physics-closed pose (same math as rebake validation).
    # Re-root desk paths onto the with_camera hierarchy for anchor math.
    class _DeskAsWorld:
        def __init__(self, stage: Usd.Stage, prefix: str):
            self._stage = stage
            self._prefix = prefix

        def GetPrimAtPath(self, path: str):
            if path.startswith("/toolbox_with_handle"):
                path = self._prefix + path.removeprefix("/toolbox_with_handle")
            return self._stage.GetPrimAtPath(path)

    proxy = _DeskAsWorld(scene, prefix)
    root_prim = proxy.GetPrimAtPath(ROOT)
    root_world = cache.GetLocalToWorldTransform(root_prim)
    body0_path = UsdPhysics.Joint(joint).GetBody0Rel().GetTargets()[0]
    body0 = scene.GetPrimAtPath(str(body0_path))
    pos0 = joint.GetAttribute("physics:localPos0").Get()
    anchor_world = cache.GetLocalToWorldTransform(body0).Transform(
        Gf.Vec3d(pos0[0], pos0[1], pos0[2])
    )
    anchor_root = root_world.GetInverse().Transform(anchor_world)

    mesh = scene.GetPrimAtPath(_world_path(prefix, f"{TOP_DRAWER_LINK}/drawer"))
    pts = UsdGeom.Mesh(mesh).GetPointsAttr().Get()
    top_z = (min(anchor_root[2] + p[2] for p in pts), max(anchor_root[2] + p[2] for p in pts))
    bottom_z = _mesh_z_range_in_root(desk, f"{STATIC_BOTTOM}/drawer_02")
    overlap = min(top_z[1], bottom_z[1]) - max(top_z[0], bottom_z[0])

    toolbox_t = cache.GetLocalToWorldTransform(scene.GetPrimAtPath(prefix)).ExtractTranslation()
    handle_local = desk.GetPrimAtPath(f"{TOP_DRAWER_LINK}/drawer_handle_top")
    h_ops = UsdGeom.Xformable(handle_local).GetOrderedXformOps()
    h_local = h_ops[0].Get() if h_ops else Gf.Vec3d(0, 0, 0)
    handle_world = anchor_world + Gf.Vec3d(h_local[0], h_local[1], h_local[2])

    print("\n[with_camera] CLOSED pose (after Play — what RL uses):")
    print(f"  toolbox world translate: ({toolbox_t[0]:.4f}, {toolbox_t[1]:.4f}, {toolbox_t[2]:.4f})")
    print(f"  top drawer Z in root:      [{top_z[0]:.4f}, {top_z[1]:.4f}]")
    print(f"  bottom static Z in root:   [{bottom_z[0]:.4f}, {bottom_z[1]:.4f}]")
    print(f"  Z overlap:                 {overlap:.4f} m  {'PASS' if overlap <= 0.001 else 'FAIL'}")
    print(
        f"  handle world (est.):       ({handle_world[0]:.4f}, {handle_world[1]:.4f}, {handle_world[2]:.4f})"
    )

    print("\n[with_camera] Viewport BEFORE Play:")
    print("  Top drawer link xform is identity — meshes may look below the bottom drawer.")
    print("  Press Play once; top drawer should snap into the upper slot (no floating).")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", default=DEFAULT_SCENE)
    parser.add_argument("--desk", default=DEFAULT_DESK)
    parser.add_argument("--flat", default=DEFAULT_FLAT)
    args = parser.parse_args()

    print("=== Desk USD rebake validation ===")
    flat = Usd.Stage.Open(args.flat)
    desk = Usd.Stage.Open(args.desk)
    if flat is None or desk is None:
        raise RuntimeError("Could not open flat/desk USD")
    _validate_desk(desk, flat)

    print("\n=== with_camera payload ===")
    _check_payload(args.scene, args.desk)
    _report_with_camera(args.scene, args.desk)

    print("\n=== Isaac Sim visual checklist ===")
    print("1. File → Open → /home/user/Desktop/with_camera.usda  (Reopen if already open)")
    print("2. Press Play ▶ — top (articulated) drawer sits in UPPER slot")
    print("3. Bottom gray knob (static_drawer_02) does NOT move")
    print("4. Shift+drag top drawer along -Y opens ~20 cm without gap/floating")
    print("5. Stop — drawer returns to closed (joint position 0)")


if __name__ == "__main__":
    main()
