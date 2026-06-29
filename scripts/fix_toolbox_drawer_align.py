"""Final drawer alignment: COM at link origin, no duplicate transforms, handle offset.

- Zero toolbox_with_handle root translate (spawn via ArticulationCfg only).
- Drawer link: identity xform, physics:centerOfMass = (0,0,0) so FrameTransformer
  rigid-body pose matches USD link / handle visuals when use_fabric=False.
- localPos0.y = 0 (closed Y), keep localPos0.z for tier Z.
- Refresh _DRAWER_HANDLE_OFFSET from knob bbox in drawer link frame.

Run:
    python scripts/fix_toolbox_drawer_align.py
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

ROOT = "/toolbox_with_handle"
DRAWER = f"{ROOT}/toolbox/toolbox/drawer"
KNOB_MESH = f"{DRAWER}/FurnitureKnob_01/Mesh"
JOINT = f"{ROOT}/toolbox/toolbox/drawer_joint"
BODY0 = f"{ROOT}/toolbox/toolbox/toolbox"
DRAWER_MESH = f"{DRAWER}/drawer"
BOTTOM_MESH = f"{BODY0}/static_drawer_02/drawer_02"


def _strip_xform_ops(prim: Usd.Prim) -> UsdGeom.Xformable:
    xf = UsdGeom.Xformable(prim)
    xf.ClearXformOpOrder()
    for attr in prim.GetAttributes():
        name = attr.GetName()
        if name.startswith("xformOp:") and attr.IsAuthored():
            prim.RemoveProperty(name)
    return xf


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


def _set_com_at_link_origin(drawer_prim: Usd.Prim) -> None:
    mass_api = UsdPhysics.MassAPI.Apply(drawer_prim)
    mass_api.CreateCenterOfMassAttr().Set(Gf.Vec3f(0.0, 0.0, 0.0))


def _zero_spawn_root(stage: Usd.Stage) -> None:
    root = stage.GetPrimAtPath(ROOT)
    _strip_xform_ops(root)
    UsdGeom.Xformable(root).AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(0.0, 0.0, 0.0))


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
        w_off = off_body.Transform(drawer_inv.Transform(w_on))
        on_ys.append(w_on[1])
        on_zs.append(w_on[2])
        off_ys.append(w_off[1])
        off_zs.append(w_off[2])

    bxf = cache.GetLocalToWorldTransform(stage.GetPrimAtPath(BOTTOM_MESH))
    bzs = [
        bxf.Transform(Gf.Vec3d(p[0], p[1], p[2]))[2]
        for p in UsdGeom.Mesh(stage.GetPrimAtPath(BOTTOM_MESH)).GetPointsAttr().Get()
    ]

    print(f"  localPos0: {lp0}")
    print(f"  root translate: {cache.GetLocalToWorldTransform(stage.GetPrimAtPath(ROOT)).ExtractTranslation()}")
    print(f"  ON drawer Y/Z: [{min(on_ys):.4f}, {max(on_ys):.4f}] / [{min(on_zs):.4f}, {max(on_zs):.4f}]")
    print(f"  OFF drawer Y/Z: [{min(off_ys):.4f}, {max(off_ys):.4f}] / [{min(off_zs):.4f}, {max(off_zs):.4f}]")
    print(f"  bottom static Z: [{min(bzs):.4f}, {max(bzs):.4f}]")

    y_delta = max(abs(min(off_ys) - min(on_ys)), abs(max(off_ys) - max(on_ys)))
    if y_delta > 0.002:
        raise RuntimeError(f"Spawn-link vs closed-body Y mismatch {y_delta:.4f} m")
    if min(off_zs) < max(bzs) - 0.005:
        raise RuntimeError("Top drawer OFF Z overlaps bottom tier")


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
    backup = USD_PATH.replace(".usda", "_pre_drawer_align.usda")
    if not os.path.isfile(backup):
        shutil.copy2(USD_PATH, backup)

    stage = Usd.Stage.Open(USD_PATH)

    _zero_spawn_root(stage)

    drawer = stage.GetPrimAtPath(DRAWER)
    _strip_xform_ops(drawer)
    _set_com_at_link_origin(drawer)

    joint = stage.GetPrimAtPath(JOINT)
    pos0 = joint.GetAttribute("physics:localPos0").Get()
    old_lp0 = Gf.Vec3d(pos0[0], pos0[1], pos0[2])
    # Keep localPos0.y = 0; closed Y alignment via mesh shift (fix_toolbox_drawer_y.py).
    new_lp0 = Gf.Vec3d(pos0[0], 0.0, pos0[2])
    if abs(old_lp0[1]) > 1e-6 or abs(old_lp0[2] - new_lp0[2]) > 1e-6:
        joint.GetAttribute("physics:localPos0").Set(Gf.Vec3f(new_lp0[0], new_lp0[1], new_lp0[2]))
        print(f"localPos0: {old_lp0} -> {new_lp0}")

    offset = _physx_handle_offset(stage)
    print(f"handle offset in PhysX body frame: {offset}")

    _validate(stage)
    stage.GetRootLayer().Save()
    print(f"Wrote {USD_PATH}")
    shutil.copy2(USD_PATH, DESK_PATH)

    _write_offset(offset)
    print("Updated environments.py _DRAWER_HANDLE_OFFSET")


if __name__ == "__main__":
    main()
