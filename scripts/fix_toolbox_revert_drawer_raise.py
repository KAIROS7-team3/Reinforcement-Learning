"""Revert drawer link translate and restore the correct knob marker offset.

The drawer-visual pass added link translate=localPos0 and changed the offset to the full
knob position in link frame. That raised both the marker and drawer mesh on Z.

Restore:
  - drawer link xform = identity
  - offset = knob_center_in_link - localPos0  (``drawer_frame`` / marker at knob)
  - drawer mesh stays in link frame (Fabric ON visual path)

Run from Reinforcement-Learning:
    /home/user/miniconda3/envs/env_isaaclab/bin/python scripts/fix_toolbox_revert_drawer_raise.py
"""

from __future__ import annotations

import os
import re
import shutil

from pxr import Gf, Usd, UsdGeom

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ASSETS = os.path.join(REPO, "assets")
USD_PATH = os.path.join(ASSETS, "toolbox_rl_flat.usda")
DESK_PATH = os.path.join(ASSETS, "toolbox_rl_desk.usda")
ENVIRONMENTS_PY = os.path.join(
    REPO, "source", "tool_transfer_bot", "assets", "environments.py"
)

ROOT = "/toolbox_with_handle"
BODY0 = f"{ROOT}/toolbox/toolbox/toolbox"
DRAWER = f"{ROOT}/toolbox/toolbox/drawer"
DRAWER_MESH = f"{DRAWER}/drawer"
KNOB_MESH = f"{DRAWER}/FurnitureKnob_01/Mesh"
HANDLE_TOP = f"{DRAWER}/drawer_handle_top"
JOINT = f"{ROOT}/toolbox/toolbox/drawer_joint"
BOTTOM_MESH = f"{BODY0}/static_drawer_02/drawer_02"
SPAWN = (0.3877008091166735, 0.56212, 0.058999998658895464)


def _strip_drawer_link_xform(stage: Usd.Stage) -> None:
    prim = stage.GetPrimAtPath(DRAWER)
    xf = UsdGeom.Xformable(prim)
    xf.ClearXformOpOrder()
    for attr in prim.GetAttributes():
        name = attr.GetName()
        if name.startswith("xformOp:") and attr.IsAuthored():
            prim.RemoveProperty(name)


def _knob_center_in_drawer_frame(stage: Usd.Stage) -> tuple[float, float, float]:
    cache = UsdGeom.XformCache()
    drawer = stage.GetPrimAtPath(DRAWER)
    knob = stage.GetPrimAtPath(KNOB_MESH)
    drawer_inv = cache.GetLocalToWorldTransform(drawer).GetInverse()
    knob_xf = cache.GetLocalToWorldTransform(knob)
    pts = UsdGeom.Mesh(knob).GetPointsAttr().Get()
    xs, ys, zs = [], [], []
    for p in pts:
        w = knob_xf.Transform(Gf.Vec3d(p[0], p[1], p[2]))
        l = drawer_inv.Transform(w)
        xs.append(l[0])
        ys.append(l[1])
        zs.append(l[2])
    return (
        (min(xs) + max(xs)) * 0.5,
        (min(ys) + max(ys)) * 0.5,
        (min(zs) + max(zs)) * 0.5,
    )


def _physx_handle_offset(stage: Usd.Stage) -> tuple[float, float, float]:
    knob = _knob_center_in_drawer_frame(stage)
    pos0 = stage.GetPrimAtPath(JOINT).GetAttribute("physics:localPos0").Get()
    return (knob[0] - pos0[0], knob[1] - pos0[1], knob[2] - pos0[2])


def _write_offset(offset: tuple[float, float, float]) -> None:
    with open(ENVIRONMENTS_PY, encoding="utf-8") as f:
        text = f.read()
    new_line = f"_DRAWER_HANDLE_OFFSET = ({offset[0]:.8g}, {offset[1]:.8g}, {offset[2]:.8g})"
    updated, n = re.subn(
        r"_DRAWER_HANDLE_OFFSET = \([^)]+\)",
        new_line,
        text,
        count=1,
    )
    if n != 1:
        raise RuntimeError("Could not update _DRAWER_HANDLE_OFFSET")
    comment = (
        "# Knob center in PhysX drawer body frame (marker good at this offset).\n"
        "# offset = knob_center_in_drawer_link - localPos0"
    )
    updated = re.sub(
        r"# Knob center[^\n]*\n(?:# [^\n]*\n)?",
        comment + "\n",
        updated,
        count=1,
    )
    with open(ENVIRONMENTS_PY, "w", encoding="utf-8") as f:
        f.write(updated)


def _set_handle_top(stage: Usd.Stage, pos: tuple[float, float, float]) -> None:
    prim = stage.GetPrimAtPath(HANDLE_TOP)
    xf = UsdGeom.Xformable(prim)
    xf.ClearXformOpOrder()
    for attr in prim.GetAttributes():
        name = attr.GetName()
        if name.startswith("xformOp:") and attr.IsAuthored():
            prim.RemoveProperty(name)
    xf.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*pos))


def _validate(stage: Usd.Stage, offset: tuple[float, float, float]) -> None:
    cache = UsdGeom.XformCache()
    spawn = Gf.Matrix4d(1.0).SetTranslateOnly(Gf.Vec3d(*SPAWN))
    lp0 = Gf.Vec3d(*stage.GetPrimAtPath(JOINT).GetAttribute("physics:localPos0").Get())
    body0 = spawn * cache.GetLocalToWorldTransform(stage.GetPrimAtPath(BODY0))
    off_body = body0 * Gf.Matrix4d(1.0).SetTranslateOnly(lp0)

    mesh_xf = spawn * cache.GetLocalToWorldTransform(stage.GetPrimAtPath(DRAWER_MESH))
    on_zs = [
        mesh_xf.Transform(Gf.Vec3d(*p))[2]
        for p in UsdGeom.Mesh(stage.GetPrimAtPath(DRAWER_MESH)).GetPointsAttr().Get()
    ]
    bxf = spawn * cache.GetLocalToWorldTransform(stage.GetPrimAtPath(BOTTOM_MESH))
    bzs = [
        bxf.Transform(Gf.Vec3d(*p))[2]
        for p in UsdGeom.Mesh(stage.GetPrimAtPath(BOTTOM_MESH)).GetPointsAttr().Get()
    ]

    marker = off_body.Transform(Gf.Vec3d(*offset))
    knob = stage.GetPrimAtPath(KNOB_MESH)
    kxf = spawn * cache.GetLocalToWorldTransform(knob)
    kws = [kxf.Transform(Gf.Vec3d(*p)) for p in UsdGeom.Mesh(knob).GetPointsAttr().Get()]
    kzs = [p[2] for p in kws]

    tier_gap = min(on_zs) - max(bzs)
    print(f"  localPos0: {lp0}")
    print(f"  Fabric ON drawer Z: [{min(on_zs):.4f}, {max(on_zs):.4f}]")
    print(f"  marker Z: {marker[2]:.4f}")
    print(f"  knob ON Z: [{min(kzs):.4f}, {max(kzs):.4f}]")
    print(f"  tier gap ON: {tier_gap:.4f} m")

    if abs(tier_gap) > 0.005:
        raise RuntimeError(f"Tier gap out of range: {tier_gap:.4f} m")
    if not (min(kzs) - 0.01 <= marker[2] <= max(kzs) + 0.01):
        raise RuntimeError("Marker Z outside knob ON range")


def main() -> None:
    if not os.path.isfile(USD_PATH):
        raise FileNotFoundError(USD_PATH)

    stage = Usd.Stage.Open(USD_PATH)
    _strip_drawer_link_xform(stage)
    print("drawer link xform: identity (removed translate)")

    offset = _physx_handle_offset(stage)
    print(f"handle offset (PhysX body frame): {offset}")
    _set_handle_top(stage, offset)
    _write_offset(offset)
    _validate(stage, offset)

    stage.GetRootLayer().Save()
    print(f"Wrote {USD_PATH}")
    shutil.copy2(USD_PATH, DESK_PATH)
    print(f"Copied -> {DESK_PATH}")


if __name__ == "__main__":
    main()
