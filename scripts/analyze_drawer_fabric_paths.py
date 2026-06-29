"""Print Fabric-ON vs PhysX-OFF static paths (USD) + handle marker offset check."""

from __future__ import annotations

import os

from pxr import Gf, Usd, UsdGeom

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
USD = os.path.join(REPO, "assets", "toolbox_rl_flat.usda")
ROOT = "/toolbox_with_handle"
PARENT = f"{ROOT}/toolbox/toolbox"
BODY0 = f"{ROOT}/toolbox/toolbox/toolbox"
DRAWER = f"{ROOT}/toolbox/toolbox/drawer"
DRAWER_MESH = f"{DRAWER}/drawer"
KNOB = f"{DRAWER}/FurnitureKnob_01/Mesh"
JOINT = f"{ROOT}/toolbox/toolbox/drawer_joint"
BOTTOM = f"{BODY0}/static_drawer_02/drawer_02"
# ArticulationCfg spawn (environments.py)
SPAWN = (0.3877008091166735, 0.56212, 0.058999998658895464)
# PhysX body-frame handle offset (environments.py)
HANDLE_OFFSET = (-5.090332e-05, -0.11836937, 0.022230724)


def mesh_bbox_world(stage: Usd.Stage, path: str, cache: UsdGeom.XformCache) -> tuple[tuple, tuple, tuple]:
    prim = stage.GetPrimAtPath(path)
    xf = cache.GetLocalToWorldTransform(prim)
    pts = UsdGeom.Mesh(prim).GetPointsAttr().Get()
    ws = [xf.Transform(Gf.Vec3d(p[0], p[1], p[2])) for p in pts]
    xs = [w[0] for w in ws]
    ys = [w[1] for w in ws]
    zs = [w[2] for w in ws]
    return (
        (min(xs), max(xs)),
        (min(ys), max(ys)),
        (min(zs), max(zs)),
    )


def main() -> None:
    stage = Usd.Stage.Open(USD)
    cache = UsdGeom.XformCache()

    # spawn offset on root
    spawn_mat = Gf.Matrix4d(1.0).SetTranslateOnly(Gf.Vec3d(*SPAWN))

    joint = stage.GetPrimAtPath(JOINT)
    pos0 = joint.GetAttribute("physics:localPos0").Get()
    local_pos0 = Gf.Vec3d(pos0[0], pos0[1], pos0[2])

    parent_world = spawn_mat * cache.GetLocalToWorldTransform(stage.GetPrimAtPath(PARENT))
    body0_world = spawn_mat * cache.GetLocalToWorldTransform(stage.GetPrimAtPath(BODY0))
    drawer_world = spawn_mat * cache.GetLocalToWorldTransform(stage.GetPrimAtPath(DRAWER))
    anchor_world = body0_world.Transform(local_pos0)
    off_body_mat = body0_world * Gf.Matrix4d(1.0).SetTranslateOnly(local_pos0)

    mesh = stage.GetPrimAtPath(DRAWER_MESH)
    mesh_xf = spawn_mat * cache.GetLocalToWorldTransform(mesh)
    pts = UsdGeom.Mesh(mesh).GetPointsAttr().Get()
    mesh_world = [mesh_xf.Transform(Gf.Vec3d(p[0], p[1], p[2])) for p in pts]

    # Fabric ON: drawer link at spawn (identity), mesh as authored in link frame
    on_ys = [p[1] for p in mesh_world]
    on_zs = [p[2] for p in mesh_world]

    # PhysX OFF: body1 origin at anchor, mesh in body1 local (same authored points)
    drawer_inv = drawer_world.GetInverse()
    off_world = []
    for p in mesh_world:
        link_local = drawer_inv.Transform(p)
        off_world.append(off_body_mat.Transform(link_local))
    off_ys = [p[1] for p in off_world]
    off_zs = [p[2] for p in off_world]

    print("localPos0:", local_pos0)
    print("anchor world Z:", anchor_world[2])
    print("drawer link world Z:", drawer_world.ExtractTranslation()[2])
    print()
    print("Fabric ON (link spawn) drawer Y:", (min(on_ys), max(on_ys)))
    print("PhysX OFF (body anchor) drawer Y:", (min(off_ys), max(off_ys)))
    print("Fabric ON drawer Z:", (min(on_zs), max(on_zs)))
    print("PhysX OFF drawer Z:", (min(off_zs), max(off_zs)))

    knob_on = mesh_bbox_world(stage, KNOB, UsdGeom.XformCache())
    # recompute knob with spawn
    cache2 = UsdGeom.XformCache()
    kxf = spawn_mat * cache2.GetLocalToWorldTransform(stage.GetPrimAtPath(KNOB))
    kpts = UsdGeom.Mesh(stage.GetPrimAtPath(KNOB)).GetPointsAttr().Get()
    kws = [kxf.Transform(Gf.Vec3d(p[0], p[1], p[2])) for p in kpts]
    print("Knob ON Y:", (min(p[1] for p in kws), max(p[1] for p in kws)))
    koff2 = []
    for p in kws:
        ll = drawer_inv.Transform(p)
        koff2.append(off_body_mat.Transform(ll))
    print("Knob OFF Y:", (min(p[1] for p in koff2), max(p[1] for p in koff2)))
    kon_zs = [p[2] for p in kws]
    koff_zs = [p[2] for p in koff2]
    print("Knob ON Z:", (min(kon_zs), max(kon_zs)))
    print("Knob OFF Z:", (min(koff_zs), max(koff_zs)))

    # Marker target = body_anchor_world + offset (PhysX OFF path)
    marker_off = off_body_mat.Transform(
        Gf.Vec3d(HANDLE_OFFSET[0], HANDLE_OFFSET[1], HANDLE_OFFSET[2])
    )
    knob_off_center = Gf.Vec3d(
        sum(p[0] for p in koff2) / len(koff2),
        sum(p[1] for p in koff2) / len(koff2),
        sum(p[2] for p in koff2) / len(koff2),
    )
    marker_on = drawer_world.Transform(
        Gf.Vec3d(HANDLE_OFFSET[0], HANDLE_OFFSET[1], HANDLE_OFFSET[2])
    )
    knob_on_center = Gf.Vec3d(
        sum(p[0] for p in kws) / len(kws),
        sum(p[1] for p in kws) / len(kws),
        sum(p[2] for p in kws) / len(kws),
    )
    print()
    print("Marker target OFF (body+offset):", tuple(round(c, 5) for c in marker_off))
    print("Knob visual OFF center:         ", tuple(round(c, 5) for c in knob_off_center))
    print("Marker target ON (link+offset):   ", tuple(round(c, 5) for c in marker_on))
    print("Knob visual ON center:          ", tuple(round(c, 5) for c in knob_on_center))
    err_off = marker_off - knob_off_center
    err_on = marker_on - knob_on_center
    print(f"|marker−knob| OFF: {err_off.GetLength():.6f} m  ON: {err_on.GetLength():.6f} m")

    bxf = spawn_mat * cache2.GetLocalToWorldTransform(stage.GetPrimAtPath(BOTTOM))
    bpts = UsdGeom.Mesh(stage.GetPrimAtPath(BOTTOM)).GetPointsAttr().Get()
    bzs = [bxf.Transform(Gf.Vec3d(p[0], p[1], p[2]))[2] for p in bpts]
    print("Bottom drawer Z:", (min(bzs), max(bzs)))


if __name__ == "__main__":
    main()
