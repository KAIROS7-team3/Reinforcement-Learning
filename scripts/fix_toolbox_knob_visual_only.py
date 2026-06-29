"""Shift only the white drawer knob visual (FurnitureKnob_01) in link frame.

After fix_toolbox_drawer_visual_only.py moved the drawer mesh/collision down by
-localPos0.z, the white knob stayed at the old height. Apply the same Z shift to
FurnitureKnob_01 only. Marker offset is unchanged.

Run from Reinforcement-Learning:
    /home/user/miniconda3/envs/env_isaaclab/bin/python scripts/fix_toolbox_knob_visual_only.py
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
DRAWER = f"{ROOT}/toolbox/toolbox/drawer"
KNOB_ROOT = f"{DRAWER}/FurnitureKnob_01"
KNOB_MESH = f"{KNOB_ROOT}/Mesh"
DRAWER_MESH = f"{DRAWER}/drawer"
JOINT = f"{ROOT}/toolbox/toolbox/drawer_joint"
SPAWN = (0.3877008091166735, 0.56212, 0.058999998658895464)


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


def _shift_translate_z(prim: Usd.Prim, dz: float) -> None:
    xf = UsdGeom.Xformable(prim)
    for op in xf.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            v = op.Get()
            if v is not None:
                op.Set(Gf.Vec3d(v[0], v[1], v[2] + dz))


def _shift_knob_z(stage: Usd.Stage, dz: float) -> None:
    knob = stage.GetPrimAtPath(KNOB_ROOT)
    for prim in Usd.PrimRange(knob):
        if prim.IsA(UsdGeom.Mesh):
            _shift_mesh_z(prim, dz)
        elif prim.IsA(UsdGeom.Xformable) and prim != knob:
            _shift_translate_z(prim, dz)


def _z_range(stage: Usd.Stage, path: str) -> tuple[float, float]:
    cache = UsdGeom.XformCache()
    spawn = Gf.Matrix4d(1.0).SetTranslateOnly(Gf.Vec3d(*SPAWN))
    prim = stage.GetPrimAtPath(path)
    xf = spawn * cache.GetLocalToWorldTransform(prim)
    zs = [xf.Transform(Gf.Vec3d(*p))[2] for p in UsdGeom.Mesh(prim).GetPointsAttr().Get()]
    return min(zs), max(zs)


def _validate(stage: Usd.Stage) -> None:
    drawer_z = _z_range(stage, DRAWER_MESH)
    knob_z = _z_range(stage, KNOB_MESH)
    overlap = min(knob_z[1], drawer_z[1]) - max(knob_z[0], drawer_z[0])
    print(f"  drawer mesh ON Z: [{drawer_z[0]:.4f}, {drawer_z[1]:.4f}]")
    print(f"  white knob ON Z:  [{knob_z[0]:.4f}, {knob_z[1]:.4f}]")
    print(f"  Z overlap: {overlap:.4f} m")
    if overlap < 0.005:
        raise RuntimeError("Knob Z does not overlap drawer mesh after shift")


def main() -> None:
    if not os.path.isfile(USD_PATH):
        raise FileNotFoundError(USD_PATH)

    stage = Usd.Stage.Open(USD_PATH)
    lp0 = stage.GetPrimAtPath(JOINT).GetAttribute("physics:localPos0").Get()
    dz = -float(lp0[2])
    if abs(dz) < 1e-6:
        print("localPos0.z is zero; nothing to shift")
        return

    z_before = _z_range(stage, KNOB_MESH)
    _shift_knob_z(stage, dz)
    z_after = _z_range(stage, KNOB_MESH)

    print(f"shifted FurnitureKnob_01 {dz:+.6f} m in link frame")
    print(f"white knob ON Z: [{z_before[0]:.4f}, {z_before[1]:.4f}] -> [{z_after[0]:.4f}, {z_after[1]:.4f}]")
    _validate(stage)

    stage.GetRootLayer().Save()
    print(f"Wrote {USD_PATH}")
    shutil.copy2(USD_PATH, DESK_PATH)
    print(f"Copied -> {DESK_PATH}")


if __name__ == "__main__":
    main()
