"""Restore top drawer (drawer_joint) to 2nd-tier Z — rebake placed it at bottom tier.

Run from repo root:
    python scripts/fix_toolbox_drawer_tier.py
"""

from __future__ import annotations

import os
import shutil

from pxr import Gf, Usd, UsdGeom

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ASSETS = os.path.join(REPO, "assets")
USD_PATH = os.path.join(ASSETS, "toolbox_rl_flat.usda")
DESK_PATH = os.path.join(ASSETS, "toolbox_rl_desk.usda")
PRE_REBAKE = os.path.join(ASSETS, "toolbox_rl_flat_pre_rebake.usda")

ROOT = "/toolbox_with_handle/toolbox"
TOP_DRAWER_LINK = f"{ROOT}/toolbox/drawer"
STATIC_BOTTOM_MESH = f"{ROOT}/toolbox/toolbox/static_drawer_02/drawer_02"
TOP_DRAWER_MESH = f"{TOP_DRAWER_LINK}/drawer"
HANDLE_TOP = f"{TOP_DRAWER_LINK}/drawer_handle_top"


def _xform_cache() -> UsdGeom.XformCache:
    return UsdGeom.XformCache()


def _mesh_z_in_root(stage: Usd.Stage, mesh_path: str) -> tuple[float, float]:
    prim = stage.GetPrimAtPath(mesh_path)
    if not prim.IsValid():
        return (0.0, 0.0)
    cache = _xform_cache()
    root_world = cache.GetLocalToWorldTransform(stage.GetPrimAtPath(ROOT))
    root_inv = root_world.GetInverse()
    mesh_world = cache.GetLocalToWorldTransform(prim)
    zs = [
        root_inv.Transform(mesh_world.Transform(Gf.Vec3d(p[0], p[1], p[2])))[2]
        for p in UsdGeom.Mesh(prim).GetPointsAttr().Get()
    ]
    return (min(zs), max(zs))


def _shift_mesh_z(mesh_prim: Usd.Prim, dz: float) -> None:
    mesh = UsdGeom.Mesh(mesh_prim)
    pts = mesh.GetPointsAttr().Get()
    shifted = [Gf.Vec3f(p[0], p[1], p[2] + dz) for p in pts]
    mesh.GetPointsAttr().Set(shifted)
    ext = mesh.GetExtentAttr().Get()
    if ext:
        mesh.GetExtentAttr().Set(
            [Gf.Vec3f(ext[0][0], ext[0][1], ext[0][2] + dz), Gf.Vec3f(ext[1][0], ext[1][1], ext[1][2] + dz)]
        )


def _shift_cube_z(cube_prim: Usd.Prim, dz: float) -> None:
    xf = UsdGeom.Xformable(cube_prim)
    ops = xf.GetOrderedXformOps()
    if ops:
        mat = ops[0].Get()
        if mat:
            t = mat.ExtractTranslation()
            mat.SetTranslateOnly(Gf.Vec3d(t[0], t[1], t[2] + dz))
            ops[0].Set(mat)


def _shift_handle_top(stage: Usd.Stage, dz: float) -> None:
    prim = stage.GetPrimAtPath(HANDLE_TOP)
    if not prim.IsValid():
        return
    xf = UsdGeom.Xformable(prim)
    for op in xf.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            v = op.Get()
            op.Set(Gf.Vec3d(v[0], v[1], v[2] + dz))


SPAWN = (0.3877, 0.56212, 0.059)


def _mesh_world_z(stage: Usd.Stage, mesh_path: str, spawn: tuple[float, float, float]) -> tuple[float, float]:
    prim = stage.GetPrimAtPath(mesh_path)
    if not prim.IsValid():
        return (0.0, 0.0)
    root = stage.GetPrimAtPath("/toolbox_with_handle")
    UsdGeom.Xformable(root).ClearXformOpOrder()
    UsdGeom.Xformable(root).AddTranslateOp().Set(Gf.Vec3d(*spawn))
    cache = _xform_cache()
    xf = cache.GetLocalToWorldTransform(prim)
    zs = [xf.Transform(Gf.Vec3d(p[0], p[1], p[2]))[2] for p in UsdGeom.Mesh(prim).GetPointsAttr().Get()]
    return (min(zs), max(zs))


def apply_tier_shift_to_stage(stage: Usd.Stage, ref_stage: Usd.Stage | None = None) -> float:
    """Shift top drawer link geometry to 2nd tier. Returns dz applied (0 if skipped)."""
    dz = _compute_shift(stage, ref_stage)
    if abs(dz) < 1e-9:
        return 0.0
    link = stage.GetPrimAtPath(TOP_DRAWER_LINK)
    for prim in Usd.PrimRange(link):
        if prim.IsA(UsdGeom.Mesh):
            _shift_mesh_z(prim, dz)
        elif prim.IsA(UsdGeom.Cube):
            _shift_cube_z(prim, dz)
    _shift_handle_top(stage, dz)
    return dz


def validate_tier(stage: Usd.Stage) -> None:    validate_tier(stage)


def _compute_shift(stage: Usd.Stage, ref_stage: Usd.Stage | None = None) -> float:
    """Match pre-rebake top drawer world Z (2nd tier) at standard spawn."""
    cur_top = _mesh_world_z(stage, TOP_DRAWER_MESH, SPAWN)
    if ref_stage is not None:
        ref_top = _mesh_world_z(ref_stage, f"{ROOT}/toolbox/drawer/drawer", SPAWN)
        if ref_top[1] > ref_top[0]:
            return ref_top[0] - cur_top[0]
    if os.path.isfile(PRE_REBAKE):
        ref_stage = Usd.Stage.Open(PRE_REBAKE)
        ref_top = _mesh_world_z(ref_stage, f"{ROOT}/toolbox/drawer/drawer", SPAWN)
        if ref_top[1] > ref_top[0]:
            return ref_top[0] - cur_top[0]

    bottom = _mesh_z_in_root(stage, STATIC_BOTTOM_MESH)
    clearance = 0.001
    cur_root = _mesh_z_in_root(stage, TOP_DRAWER_MESH)
    return bottom[1] + clearance - cur_root[0]


def _validate(stage: Usd.Stage) -> None:
    top = _mesh_z_in_root(stage, TOP_DRAWER_MESH)
    bottom = _mesh_z_in_root(stage, STATIC_BOTTOM_MESH)
    overlap = min(top[1], bottom[1]) - max(top[0], bottom[0])
    print(f"  top drawer Z in root: [{top[0]:.4f}, {top[1]:.4f}]")
    print(f"  bottom drawer Z in root: [{bottom[0]:.4f}, {bottom[1]:.4f}]")
    print(f"  Z overlap: {overlap:.4f} m (expect <= 0)")
    if overlap > 0.001:
        raise RuntimeError(f"Top/bottom drawer overlap {overlap:.4f} m after tier fix")
    if top[0] < bottom[1] - 0.005:
        raise RuntimeError(
            f"Top drawer min Z {top[0]:.4f} should be above bottom max Z {bottom[1]:.4f}"
        )


def main() -> None:
    if not os.path.isfile(USD_PATH):
        raise FileNotFoundError(USD_PATH)

    backup = USD_PATH.replace(".usda", "_pre_tier_fix.usda")
    if not os.path.isfile(backup):
        shutil.copy2(USD_PATH, backup)

    stage = Usd.Stage.Open(USD_PATH)
    dz = apply_tier_shift_to_stage(stage)
    print(f"Shifting top drawer link geometry by dz={dz:.6f} m")
    stage.GetRootLayer().Save()
    print(f"Wrote {USD_PATH}")

    shutil.copy2(USD_PATH, DESK_PATH)
    print(f"Copied -> {DESK_PATH}")


if __name__ == "__main__":
    main()
