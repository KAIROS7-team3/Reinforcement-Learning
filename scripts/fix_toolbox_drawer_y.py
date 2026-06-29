"""Shift top drawer mesh +Y in link frame; revert localPos0.y to 0.

localPos0.y moves PhysX body (marker) but Fabric OFF USD visuals stay on link xform.
Rebake-free fix: translate geometry in drawer link frame so visuals and physics align.

Run:
    python scripts/fix_toolbox_drawer_y.py
"""

from __future__ import annotations

import os
import re
import shutil

from pxr import Gf, Usd, UsdGeom, UsdPhysics

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ASSETS = os.path.join(REPO, "assets")
USD_PATH = os.path.join(ASSETS, "toolbox_rl_flat.usda")
DESK_PATH = os.path.join(ASSETS, "toolbox_rl_desk.usda")
ENVIRONMENTS_PY = os.path.join(
    REPO, "source", "tool_transfer_bot", "assets", "environments.py"
)
SPAWN = (0.3877008091166735, 0.56212, 0.058999998658895464)

ROOT = "/toolbox_with_handle"
BODY0 = f"{ROOT}/toolbox/toolbox/toolbox"
DRAWER = f"{ROOT}/toolbox/toolbox/drawer"
KNOB_TOP = f"{DRAWER}/FurnitureKnob_01/Mesh"
KNOB_BOTTOM = f"{BODY0}/static_drawer_02/FurnitureKnob_02/Mesh"
DRAWER_MESH = f"{DRAWER}/drawer"
JOINT = f"{ROOT}/toolbox/toolbox/drawer_joint"
HANDLE_TOP = f"{DRAWER}/drawer_handle_top"

# Closed Y gap: top knob vs bottom tier (with localPos0.y=0).
Y_SHIFT = 0.08774971


def _update_mesh_extent(mesh_prim: Usd.Prim) -> None:
    mesh = UsdGeom.Mesh(mesh_prim)
    pts = mesh.GetPointsAttr().Get()
    if not pts:
        return
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    zs = [p[2] for p in pts]
    mesh.GetExtentAttr().Set(
        [Gf.Vec3f(min(xs), min(ys), min(zs)), Gf.Vec3f(max(xs), max(ys), max(zs))]
    )


def _shift_mesh_y(mesh_prim: Usd.Prim, dy: float) -> None:
    if not mesh_prim.IsA(UsdGeom.Mesh):
        return
    mesh = UsdGeom.Mesh(mesh_prim)
    pts = mesh.GetPointsAttr().Get()
    if not pts:
        return
    shifted = [Gf.Vec3f(p[0], p[1] + dy, p[2]) for p in pts]
    mesh.GetPointsAttr().Set(shifted)
    _update_mesh_extent(mesh_prim)


def _shift_cube_y(cube_prim: Usd.Prim, dy: float) -> None:
    if not cube_prim.IsA(UsdGeom.Cube):
        return
    xf = UsdGeom.Xformable(cube_prim)
    op = xf.GetOrderedXformOps()
    if not op:
        return
    mat = op[0].Get()
    if mat is None:
        return
    mat = Gf.Matrix4d(mat)
    t = mat.ExtractTranslation()
    mat.SetTranslateOnly(Gf.Vec3d(t[0], t[1] + dy, t[2]))
    op[0].Set(mat)


def _shift_translate_xform(prim: Usd.Prim, dy: float) -> None:
    xf = UsdGeom.Xformable(prim)
    for op in xf.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            v = op.Get()
            if v is not None:
                op.Set(Gf.Vec3d(v[0], v[1] + dy, v[2]))


def _shift_drawer_geometry(stage: Usd.Stage, dy: float) -> None:
    drawer = stage.GetPrimAtPath(DRAWER)
    for prim in Usd.PrimRange(drawer):
        if prim.IsA(UsdGeom.Mesh):
            _shift_mesh_y(prim, dy)
        elif prim.IsA(UsdGeom.Cube):
            _shift_cube_y(prim, dy)
        elif prim.IsA(UsdGeom.Xformable) and prim != drawer:
            _shift_translate_xform(prim, dy)


def _knob_center_in_drawer_frame(stage: Usd.Stage) -> tuple[float, float, float]:
    cache = UsdGeom.XformCache()
    drawer = stage.GetPrimAtPath(DRAWER)
    knob = stage.GetPrimAtPath(KNOB_TOP)
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


def _visual_knob_y(stage: Usd.Stage) -> tuple[float, float]:
    """Drawer mesh world Y (USD link path, spawn applied)."""
    cache = UsdGeom.XformCache()
    spawn = Gf.Matrix4d(1.0).SetTranslateOnly(Gf.Vec3d(*SPAWN))
    xf = spawn * cache.GetLocalToWorldTransform(stage.GetPrimAtPath(KNOB_TOP))
    ys = [
        xf.Transform(Gf.Vec3d(p[0], p[1], p[2]))[1]
        for p in UsdGeom.Mesh(stage.GetPrimAtPath(KNOB_TOP)).GetPointsAttr().Get()
    ]
    return min(ys), max(ys)


def _bottom_knob_y(stage: Usd.Stage) -> tuple[float, float]:
    cache = UsdGeom.XformCache()
    spawn = Gf.Matrix4d(1.0).SetTranslateOnly(Gf.Vec3d(*SPAWN))
    xf = spawn * cache.GetLocalToWorldTransform(stage.GetPrimAtPath(KNOB_BOTTOM))
    ys = [
        xf.Transform(Gf.Vec3d(p[0], p[1], p[2]))[1]
        for p in UsdGeom.Mesh(stage.GetPrimAtPath(KNOB_BOTTOM)).GetPointsAttr().Get()
    ]
    return min(ys), max(ys)


def _closed_mesh_z(stage: Usd.Stage) -> tuple[float, float]:
    cache = UsdGeom.XformCache()
    spawn = Gf.Matrix4d(1.0).SetTranslateOnly(Gf.Vec3d(*SPAWN))
    joint = stage.GetPrimAtPath(JOINT)
    pos0 = joint.GetAttribute("physics:localPos0").Get()
    lp0 = Gf.Vec3d(pos0[0], pos0[1], pos0[2])
    body0_w = spawn * cache.GetLocalToWorldTransform(stage.GetPrimAtPath(BODY0))
    closed_origin = body0_w * Gf.Matrix4d(1.0).SetTranslateOnly(lp0)
    drawer_w = spawn * cache.GetLocalToWorldTransform(stage.GetPrimAtPath(DRAWER))
    drawer_inv = drawer_w.GetInverse()
    mesh_prim = stage.GetPrimAtPath(DRAWER_MESH)
    mesh_xf = spawn * cache.GetLocalToWorldTransform(mesh_prim)
    zs = []
    for p in UsdGeom.Mesh(mesh_prim).GetPointsAttr().Get():
        mw = mesh_xf.Transform(Gf.Vec3d(p[0], p[1], p[2]))
        ll = drawer_inv.Transform(mw)
        zs.append(closed_origin.Transform(ll)[2])
    return min(zs), max(zs)


def _physx_handle_offset(stage: Usd.Stage) -> tuple[float, float, float]:
    """Knob center in PhysX body frame (joint anchor = body0 + localPos0)."""
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
    with open(ENVIRONMENTS_PY, "w", encoding="utf-8") as f:
        f.write(updated)


def main() -> None:
    backup = USD_PATH.replace(".usda", "_pre_drawer_y_mesh.usda")
    if not os.path.isfile(backup):
        shutil.copy2(USD_PATH, backup)

    stage = Usd.Stage.Open(USD_PATH)
    joint = stage.GetPrimAtPath(JOINT)
    pos0 = joint.GetAttribute("physics:localPos0").Get()
    old_lp0 = Gf.Vec3d(pos0[0], pos0[1], pos0[2])

    dy = old_lp0[1] if abs(old_lp0[1]) > 1e-6 else Y_SHIFT
    if abs(dy) > 1e-6:
        _shift_drawer_geometry(stage, dy)
        print(f"shifted drawer link geometry by dy={dy:.6f} m")

    new_lp0 = Gf.Vec3f(pos0[0], 0.0, pos0[2])
    joint.GetAttribute("physics:localPos0").Set(new_lp0)
    print(f"localPos0: {old_lp0} -> ({new_lp0[0]}, {new_lp0[1]}, {new_lp0[2]})")

    top_y = _visual_knob_y(stage)
    bottom_y = _bottom_knob_y(stage)
    mesh_z = _closed_mesh_z(stage)
    print(f"top knob Y (visual):  {top_y}")
    print(f"bottom knob Y:        {bottom_y}")
    print(f"top drawer Z (closed): {mesh_z}")

    y_err = max(abs(top_y[0] - bottom_y[0]), abs(top_y[1] - bottom_y[1]))
    if y_err > 0.004:
        raise RuntimeError(f"Top/bottom knob Y mismatch {y_err:.4f} m after mesh shift")

    offset = _physx_handle_offset(stage)
    print(f"handle offset in PhysX body frame: {offset}")
    _write_offset(offset)

    stage.GetRootLayer().Save()
    shutil.copy2(USD_PATH, DESK_PATH)
    print(f"Wrote {USD_PATH}")


if __name__ == "__main__":
    main()
