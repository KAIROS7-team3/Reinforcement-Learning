"""Shift only drawer visuals down in link frame (keep marker/knob unchanged).

Fabric ON can render the drawer mesh on the PhysX body path (body0 + localPos0) while the
handle marker stays at the lower tier. Rebake drawer geometry by -localPos0.z in the link
frame so the body-path visual aligns with the marker.

Does NOT change drawer link xform or _DRAWER_HANDLE_OFFSET.

Run from Reinforcement-Learning:
    /home/user/miniconda3/envs/env_isaaclab/bin/python scripts/fix_toolbox_drawer_visual_only.py
"""

from __future__ import annotations

import os
import shutil

from pxr import Gf, Usd, UsdGeom

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ASSETS = os.path.join(REPO, "assets")
USD_PATH = os.path.join(ASSETS, "toolbox_rl_flat.usda")
DESK_PATH = os.path.join(ASSETS, "toolbox_rl_desk.usda")

ROOT = "/toolbox_with_handle"
BODY0 = f"{ROOT}/toolbox/toolbox/toolbox"
DRAWER = f"{ROOT}/toolbox/toolbox/drawer"
DRAWER_MESH = f"{DRAWER}/drawer"
KNOB_MESH = f"{DRAWER}/FurnitureKnob_01/Mesh"
JOINT = f"{ROOT}/toolbox/toolbox/drawer_joint"
SPAWN = (0.3877008091166735, 0.56212, 0.058999998658895464)
HANDLE_OFFSET = (-5.090332e-05, -0.11836937, 0.022230722)

SKIP_PREFIXES = (
    f"{DRAWER}/FurnitureKnob_01",
    f"{DRAWER}/drawer_handle_top",
)


def _should_skip(prim: Usd.Prim) -> bool:
    path = str(prim.GetPath())
    return any(path.startswith(prefix) for prefix in SKIP_PREFIXES)


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
    ops = xf.GetOrderedXformOps()
    if not ops:
        return
    mat = ops[0].Get()
    if mat is None:
        return
    mat = Gf.Matrix4d(mat)
    t = mat.ExtractTranslation()
    mat.SetTranslateOnly(Gf.Vec3d(t[0], t[1], t[2] + dz))
    ops[0].Set(mat)


def _shift_translate_z(prim: Usd.Prim, dz: float) -> None:
    xf = UsdGeom.Xformable(prim)
    for op in xf.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            v = op.Get()
            if v is not None:
                op.Set(Gf.Vec3d(v[0], v[1], v[2] + dz))


def _shift_drawer_visuals_z(stage: Usd.Stage, dz: float) -> None:
    drawer = stage.GetPrimAtPath(DRAWER)
    for prim in Usd.PrimRange(drawer):
        if prim == drawer or _should_skip(prim):
            continue
        if prim.IsA(UsdGeom.Mesh):
            _shift_mesh_z(prim, dz)
        elif prim.IsA(UsdGeom.Cube):
            _shift_cube_z(prim, dz)
        elif prim.IsA(UsdGeom.Xformable):
            _shift_translate_z(prim, dz)


def _validate(stage: Usd.Stage) -> None:
    cache = UsdGeom.XformCache()
    spawn = Gf.Matrix4d(1.0).SetTranslateOnly(Gf.Vec3d(*SPAWN))
    lp0 = Gf.Vec3d(*stage.GetPrimAtPath(JOINT).GetAttribute("physics:localPos0").Get())
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

    marker = off_body.Transform(Gf.Vec3d(*HANDLE_OFFSET))
    knob = stage.GetPrimAtPath(KNOB_MESH)
    kxf = spawn * cache.GetLocalToWorldTransform(knob)
    kzs = [kxf.Transform(Gf.Vec3d(*p))[2] for p in UsdGeom.Mesh(knob).GetPointsAttr().Get()]

    print(f"  localPos0.z: {lp0[2]:.6f}")
    print(f"  marker Z: {marker[2]:.4f}")
    print(f"  knob ON Z: [{min(kzs):.4f}, {max(kzs):.4f}]")
    print(f"  drawer mesh ON Z: [{min(on_zs):.4f}, {max(on_zs):.4f}]")
    print(f"  drawer mesh OFF (body path) Z: [{min(off_zs):.4f}, {max(off_zs):.4f}]")

    if not (min(off_zs) - 0.02 <= marker[2] <= max(off_zs) + 0.02):
        raise RuntimeError("Body-path drawer mesh does not cover marker Z")
    if not (min(kzs) - 0.01 <= marker[2] <= max(kzs) + 0.01):
        raise RuntimeError("Marker Z outside knob ON range (offset must stay unchanged)")


def main() -> None:
    if not os.path.isfile(USD_PATH):
        raise FileNotFoundError(USD_PATH)

    stage = Usd.Stage.Open(USD_PATH)
    lp0 = stage.GetPrimAtPath(JOINT).GetAttribute("physics:localPos0").Get()
    dz = -float(lp0[2])
    if abs(dz) < 1e-6:
        print("localPos0.z is zero; nothing to shift")
        return

    cache = UsdGeom.XformCache()
    spawn = Gf.Matrix4d(1.0).SetTranslateOnly(Gf.Vec3d(*SPAWN))
    mesh_xf = spawn * cache.GetLocalToWorldTransform(stage.GetPrimAtPath(DRAWER_MESH))
    z_before = [
        mesh_xf.Transform(Gf.Vec3d(*p))[2]
        for p in UsdGeom.Mesh(stage.GetPrimAtPath(DRAWER_MESH)).GetPointsAttr().Get()
    ]

    _shift_drawer_visuals_z(stage, dz)
    print(f"shifted drawer visuals (not knob/marker) {dz:+.6f} m in link frame")

    cache = UsdGeom.XformCache()
    mesh_xf = spawn * cache.GetLocalToWorldTransform(stage.GetPrimAtPath(DRAWER_MESH))
    z_after = [
        mesh_xf.Transform(Gf.Vec3d(*p))[2]
        for p in UsdGeom.Mesh(stage.GetPrimAtPath(DRAWER_MESH)).GetPointsAttr().Get()
    ]
    print(
        f"drawer mesh ON Z: [{min(z_before):.4f}, {max(z_before):.4f}] "
        f"-> [{min(z_after):.4f}, {max(z_after):.4f}]"
    )
    _validate(stage)

    stage.GetRootLayer().Save()
    print(f"Wrote {USD_PATH}")
    shutil.copy2(USD_PATH, DESK_PATH)
    print(f"Copied -> {DESK_PATH}")


if __name__ == "__main__":
    main()
