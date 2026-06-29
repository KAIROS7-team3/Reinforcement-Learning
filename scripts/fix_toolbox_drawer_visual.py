"""Move Fabric ON drawer visuals to the PhysX body tier (match knob marker).

Fabric ON renders the drawer link from USD while the marker follows the PhysX body at
body0 + localPos0. Apply localPos0 on the drawer link for visuals and set the
``drawer_frame`` offset to the knob center in link frame (body origin + offset = anchor + knob).

Run from Reinforcement-Learning:
    /home/user/miniconda3/envs/env_isaaclab/bin/python scripts/fix_toolbox_drawer_visual.py
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
SPAWN = (0.3877008091166735, 0.56212, 0.058999998658895464)


def _set_drawer_link_translate(stage: Usd.Stage, translate: Gf.Vec3d) -> None:
    prim = stage.GetPrimAtPath(DRAWER)
    xf = UsdGeom.Xformable(prim)
    xf.ClearXformOpOrder()
    for attr in prim.GetAttributes():
        name = attr.GetName()
        if name.startswith("xformOp:") and attr.IsAuthored():
            prim.RemoveProperty(name)
    xf.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(translate)


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
        "# Knob center in drawer link frame; with link translate=localPos0, "
        "PhysX body offset reaches the knob.\n"
        "# offset = knob_center_in_drawer_link"
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
    joint = stage.GetPrimAtPath(JOINT)
    lp0 = Gf.Vec3d(*joint.GetAttribute("physics:localPos0").Get())
    body0 = spawn * cache.GetLocalToWorldTransform(stage.GetPrimAtPath(BODY0))
    drawer = spawn * cache.GetLocalToWorldTransform(stage.GetPrimAtPath(DRAWER))
    off_body = body0 * Gf.Matrix4d(1.0).SetTranslateOnly(lp0)
    drawer_inv = drawer.GetInverse()

    mesh_prim = stage.GetPrimAtPath(DRAWER_MESH)
    mesh_xf = spawn * cache.GetLocalToWorldTransform(mesh_prim)
    on_zs, off_zs = [], []
    for p in UsdGeom.Mesh(mesh_prim).GetPointsAttr().Get():
        w_on = mesh_xf.Transform(Gf.Vec3d(*p))
        on_zs.append(w_on[2])
        ll = drawer_inv.Transform(w_on)
        off_zs.append(off_body.Transform(ll)[2])

    marker = off_body.Transform(Gf.Vec3d(*offset))
    knob = stage.GetPrimAtPath(KNOB_MESH)
    kxf = spawn * cache.GetLocalToWorldTransform(knob)
    kws = [kxf.Transform(Gf.Vec3d(*p)) for p in UsdGeom.Mesh(knob).GetPointsAttr().Get()]
    xs = [p[0] for p in kws]
    ys = [p[1] for p in kws]
    zs = [p[2] for p in kws]
    kc = Gf.Vec3d((min(xs) + max(xs)) * 0.5, (min(ys) + max(ys)) * 0.5, (min(zs) + max(zs)) * 0.5)

    print(f"  localPos0: {lp0}")
    print(f"  Fabric ON drawer Z: [{min(on_zs):.4f}, {max(on_zs):.4f}]")
    print(f"  PhysX OFF drawer Z: [{min(off_zs):.4f}, {max(off_zs):.4f}]")
    print(f"  marker/knob ON Z: {marker[2]:.4f} / {kc[2]:.4f}")

    z_delta = max(abs(min(off_zs) - min(on_zs)), abs(max(off_zs) - max(on_zs)))
    if z_delta > 0.002:
        raise RuntimeError(f"ON/OFF drawer Z mismatch {z_delta:.4f} m")
    if abs(marker[2] - kc[2]) > 0.002 or abs(marker[1] - kc[1]) > 0.002:
        raise RuntimeError("Marker does not match knob ON position")


def main() -> None:
    if not os.path.isfile(USD_PATH):
        raise FileNotFoundError(USD_PATH)

    stage = Usd.Stage.Open(USD_PATH)
    joint = stage.GetPrimAtPath(JOINT)
    lp0 = Gf.Vec3d(*joint.GetAttribute("physics:localPos0").Get())

    if abs(lp0[1]) > 1e-4:
        raise RuntimeError(
            f"localPos0.y={lp0[1]:.6f}; link translate would double-apply Y on Fabric OFF"
        )

    cache = UsdGeom.XformCache()
    spawn = Gf.Matrix4d(1.0).SetTranslateOnly(Gf.Vec3d(*SPAWN))
    mesh_prim = stage.GetPrimAtPath(DRAWER_MESH)
    z_before = [
        (spawn * cache.GetLocalToWorldTransform(mesh_prim)).Transform(Gf.Vec3d(*p))[2]
        for p in UsdGeom.Mesh(mesh_prim).GetPointsAttr().Get()
    ]

    _set_drawer_link_translate(stage, lp0)
    offset = _knob_center_in_drawer_frame(stage)
    _set_handle_top(stage, offset)
    _write_offset(offset)

    cache = UsdGeom.XformCache()
    mesh_xf = spawn * cache.GetLocalToWorldTransform(mesh_prim)
    z_after = [
        mesh_xf.Transform(Gf.Vec3d(*p))[2]
        for p in UsdGeom.Mesh(mesh_prim).GetPointsAttr().Get()
    ]

    print(f"drawer link translate: {lp0}")
    print(f"handle offset (link frame): {offset}")
    print(
        f"drawer mesh ON Z: [{min(z_before):.4f}, {max(z_before):.4f}] "
        f"-> [{min(z_after):.4f}, {max(z_after):.4f}]"
    )
    _validate(stage, offset)

    stage.GetRootLayer().Save()
    print(f"Wrote {USD_PATH}")
    shutil.copy2(USD_PATH, DESK_PATH)
    print(f"Copied -> {DESK_PATH}")


if __name__ == "__main__":
    main()
