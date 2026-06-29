"""Align Fabric ON visuals with PhysX OFF without drawer-link translate.

Wrong approach (reverted here): drawer link xformOp:translate = localPos0 doubles the
tier offset on Fabric OFF (PhysX already applies localPos0).

Correct approach: keep drawer link at identity, zero localPos0.z, rebake tier height
into drawer geometry (+Z in link frame). Fabric ON (USD link) and OFF (PhysX body at
body0) then share the same mesh in link frame.

Run:
    python scripts/fix_toolbox_fabric_align.py
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
PARENT = f"{ROOT}/toolbox/toolbox"
BODY0 = f"{ROOT}/toolbox/toolbox/toolbox"
DRAWER = f"{ROOT}/toolbox/toolbox/drawer"
DRAWER_MESH = f"{DRAWER}/drawer"
KNOB_MESH = f"{DRAWER}/FurnitureKnob_01/Mesh"
HANDLE_TOP = f"{DRAWER}/drawer_handle_top"
JOINT = f"{ROOT}/toolbox/toolbox/drawer_joint"
BOTTOM_MESH = f"{BODY0}/static_drawer_02/drawer_02"


def _strip_xform_ops(prim: Usd.Prim) -> UsdGeom.Xformable:
    xf = UsdGeom.Xformable(prim)
    xf.ClearXformOpOrder()
    for attr in prim.GetAttributes():
        name = attr.GetName()
        if name.startswith("xformOp:") and attr.IsAuthored():
            prim.RemoveProperty(name)
    return xf


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


def _shift_mesh_z(mesh_prim: Usd.Prim, dz: float) -> None:
    if not mesh_prim.IsA(UsdGeom.Mesh):
        return
    mesh = UsdGeom.Mesh(mesh_prim)
    pts = mesh.GetPointsAttr().Get()
    if not pts:
        return
    mesh.GetPointsAttr().Set([Gf.Vec3f(p[0], p[1], p[2] + dz) for p in pts])
    _update_mesh_extent(mesh_prim)


def _shift_cube_z(cube_prim: Usd.Prim, dz: float) -> None:
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
    mat.SetTranslateOnly(Gf.Vec3d(t[0], t[1], t[2] + dz))
    op[0].Set(mat)


def _shift_translate_z(prim: Usd.Prim, dz: float) -> None:
    xf = UsdGeom.Xformable(prim)
    for op in xf.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            v = op.Get()
            if v is not None:
                op.Set(Gf.Vec3d(v[0], v[1], v[2] + dz))


def _shift_drawer_geometry_z(stage: Usd.Stage, dz: float) -> None:
    drawer = stage.GetPrimAtPath(DRAWER)
    for prim in Usd.PrimRange(drawer):
        if prim.IsA(UsdGeom.Mesh):
            _shift_mesh_z(prim, dz)
        elif prim.IsA(UsdGeom.Cube):
            _shift_cube_z(prim, dz)
        elif prim.IsA(UsdGeom.Xformable) and prim != drawer:
            _shift_translate_z(prim, dz)


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


def _validate(stage: Usd.Stage) -> None:
    cache = UsdGeom.XformCache()
    body0_world = cache.GetLocalToWorldTransform(stage.GetPrimAtPath(BODY0))
    drawer_world = cache.GetLocalToWorldTransform(stage.GetPrimAtPath(DRAWER))

    joint = stage.GetPrimAtPath(JOINT)
    pos0 = joint.GetAttribute("physics:localPos0").Get()
    lp0 = Gf.Vec3d(pos0[0], pos0[1], pos0[2])
    off_body = body0_world * Gf.Matrix4d(1.0).SetTranslateOnly(lp0)

    mesh_prim = stage.GetPrimAtPath(DRAWER_MESH)
    mesh_xf = cache.GetLocalToWorldTransform(mesh_prim)
    drawer_inv = drawer_world.GetInverse()

    on_ys, on_zs, off_ys, off_zs = [], [], [], []
    for p in UsdGeom.Mesh(mesh_prim).GetPointsAttr().Get():
        w_on = mesh_xf.Transform(Gf.Vec3d(p[0], p[1], p[2]))
        ll = drawer_inv.Transform(w_on)
        w_off = off_body.Transform(ll)
        on_ys.append(w_on[1])
        on_zs.append(w_on[2])
        off_ys.append(w_off[1])
        off_zs.append(w_off[2])

    bottom_zs = []
    bxf = cache.GetLocalToWorldTransform(stage.GetPrimAtPath(BOTTOM_MESH))
    for p in UsdGeom.Mesh(stage.GetPrimAtPath(BOTTOM_MESH)).GetPointsAttr().Get():
        bottom_zs.append(bxf.Transform(Gf.Vec3d(p[0], p[1], p[2]))[2])

    print(f"  localPos0: {lp0}")
    print(f"  ON drawer Y/Z: [{min(on_ys):.4f}, {max(on_ys):.4f}] / [{min(on_zs):.4f}, {max(on_zs):.4f}]")
    print(f"  OFF drawer Y/Z: [{min(off_ys):.4f}, {max(off_ys):.4f}] / [{min(off_zs):.4f}, {max(off_zs):.4f}]")
    print(f"  bottom static Z: [{min(bottom_zs):.4f}, {max(bottom_zs):.4f}]")

    y_delta = max(abs(min(off_ys) - min(on_ys)), abs(max(off_ys) - max(on_ys)))
    z_delta = max(abs(min(off_zs) - min(on_zs)), abs(max(off_zs) - max(on_zs)))
    if y_delta > 0.002:
        raise RuntimeError(f"ON/OFF drawer Y mismatch {y_delta:.4f} m")
    if z_delta > 0.002:
        raise RuntimeError(f"ON/OFF drawer Z mismatch {z_delta:.4f} m")
    if min(off_zs) < max(bottom_zs) - 0.005:
        raise RuntimeError("Top drawer Z overlaps bottom tier")


def main() -> None:
    if not os.path.isfile(USD_PATH):
        raise FileNotFoundError(USD_PATH)

    backup = USD_PATH.replace(".usda", "_pre_fabric_align.usda")
    if not os.path.isfile(backup):
        shutil.copy2(USD_PATH, backup)

    stage = Usd.Stage.Open(USD_PATH)
    joint = stage.GetPrimAtPath(JOINT)
    pos0 = joint.GetAttribute("physics:localPos0").Get()
    old_lp0 = Gf.Vec3d(pos0[0], pos0[1], pos0[2])

    # Remove drawer link translate (breaks Fabric OFF when combined with localPos0).
    _strip_xform_ops(stage.GetPrimAtPath(DRAWER))
    print("drawer link xform: identity (no translate)")

    dz = old_lp0[2]
    new_lp0 = Gf.Vec3f(old_lp0[0], 0.0, 0.0)
    if abs(dz) > 1e-6:
        _shift_drawer_geometry_z(stage, dz)
        print(f"shifted drawer geometry +Z by {dz:.6f} m in link frame")
    joint.GetAttribute("physics:localPos0").Set(new_lp0)
    print(f"drawer_joint localPos0: {old_lp0} -> {new_lp0}")

    offset = _physx_handle_offset(stage)
    print(f"handle offset (PhysX body frame): {offset}")
    _set_handle_top(stage, offset)
    _write_offset(offset)

    _validate(stage)
    stage.GetRootLayer().Save()
    print(f"Wrote {USD_PATH}")
    shutil.copy2(USD_PATH, DESK_PATH)
    print(f"Copied -> {DESK_PATH}")


if __name__ == "__main__":
    main()
