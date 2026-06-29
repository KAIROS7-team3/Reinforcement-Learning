"""Rebake toolbox USD: link origin = visual = collision (meters, identity xforms).

Single active drawer (top / drawer_joint) for RL — bottom drawer merged as static geometry.
Run from repo root:
    python scripts/fix_toolbox_xform.py
"""

from __future__ import annotations

import os
import shutil
import sys

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

_SCRIPTS = os.path.dirname(os.path.abspath(__file__))
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)
from fix_toolbox_drawer_tier import apply_tier_shift_to_stage, validate_tier

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ASSETS = os.path.join(REPO, "assets")
SRC = os.path.join(ASSETS, "toolbox_rl_flat.usda")
DST_FLAT = SRC
DST_DESK = os.path.join(ASSETS, "toolbox_rl_desk.usda")
ROOT = "/toolbox_with_handle"
ROOT_LINK = f"{ROOT}/toolbox/toolbox/toolbox"
TOP_DRAWER_LINK = f"{ROOT}/toolbox/toolbox/toolbox/drawer"
BOTTOM_DRAWER_LINK = f"{ROOT}/toolbox/toolbox/drawer_02"
BOTTOM_JOINT = f"{ROOT}/toolbox/toolbox/drawer_02_joint"
TOP_JOINT = f"{ROOT}/toolbox/toolbox/toolbox/drawer_joint"
HANDLE_TOP = f"{TOP_DRAWER_LINK}/drawer_handle_top"
STATIC_BOTTOM = f"{ROOT_LINK}/static_drawer_02"

LINK_PATHS = [
    ROOT_LINK,
    TOP_DRAWER_LINK,
    BOTTOM_DRAWER_LINK,
    f"{ROOT}/toolbox/toolbox/toolbox/handle",  # toolbox top carry handle (NOT drawer knob)
]
ACTIVE_DRAWER_LINKS = [TOP_DRAWER_LINK]


def _strip_xform_ops(prim: Usd.Prim) -> UsdGeom.Xformable:
    xf = UsdGeom.Xformable(prim)
    xf.ClearXformOpOrder()
    for attr in prim.GetAttributes():
        name = attr.GetName()
        if name.startswith("xformOp:") and attr.IsAuthored():
            prim.RemoveProperty(name)
    return xf


def _set_identity_xform(prim: Usd.Prim) -> None:
    xf = _strip_xform_ops(prim)
    xf.AddTransformOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Matrix4d(1.0))


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


def _mesh_world_points_in_root(mesh_prim: Usd.Prim, cache: UsdGeom.XformCache, root_world: Gf.Matrix4d) -> list[Gf.Vec3f]:
    mesh = UsdGeom.Mesh(mesh_prim)
    pts = mesh.GetPointsAttr().Get()
    if not pts:
        return []
    mesh_world = cache.GetLocalToWorldTransform(mesh_prim)
    root_inv = root_world.GetInverse()
    out = []
    for p in pts:
        w = mesh_world.Transform(Gf.Vec3d(p[0], p[1], p[2]))
        r = root_inv.Transform(w)
        out.append(Gf.Vec3f(r[0], r[1], r[2]))
    return out


def _root_to_world(root_world: Gf.Matrix4d, point_in_root: Gf.Vec3d) -> Gf.Vec3d:
    return root_world.Transform(point_in_root)


def _world_to_body_local(body_prim: Usd.Prim, cache: UsdGeom.XformCache, point_world: Gf.Vec3d) -> Gf.Vec3f:
    body_inv = cache.GetLocalToWorldTransform(body_prim).GetInverse()
    local = body_inv.Transform(point_world)
    return Gf.Vec3f(local[0], local[1], local[2])


def _bake_mesh_at_joint_anchor(
    mesh_prim: Usd.Prim,
    cache: UsdGeom.XformCache,
    root_world: Gf.Matrix4d,
    anchor_root: Gf.Vec3d,
) -> None:
    """Bake mesh vertices into drawer link frame (meters), origin at closed joint anchor."""
    baked = []
    for p in _mesh_world_points_in_root(mesh_prim, cache, root_world):
        baked.append(
            Gf.Vec3f(
                p[0] - anchor_root[0],
                p[1] - anchor_root[1],
                p[2] - anchor_root[2],
            )
        )
    _write_mesh_points(mesh_prim, baked)


def _write_mesh_points(mesh_prim: Usd.Prim, points: list[Gf.Vec3f]) -> None:
    mesh = UsdGeom.Mesh(mesh_prim)
    mesh.GetPointsAttr().Set(points)
    _update_mesh_extent(mesh_prim)
    _set_identity_xform(mesh_prim)


def _mesh_bbox_center_local(mesh_prim: Usd.Prim) -> tuple[float, float, float]:
    mesh = UsdGeom.Mesh(mesh_prim)
    pts = mesh.GetPointsAttr().Get()
    if not pts:
        return (0.0, 0.0, 0.0)
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    zs = [p[2] for p in pts]
    return ((min(xs) + max(xs)) * 0.5, (min(ys) + max(ys)) * 0.5, (min(zs) + max(zs)) * 0.5)


def _snapshot_cavity_cubes(stage: Usd.Stage) -> list[tuple[str, Gf.Matrix4d, str]]:
    cache = _xform_cache()
    snapshots: list[tuple[str, Gf.Matrix4d, str]] = []
    for link_path in ACTIVE_DRAWER_LINKS:
        link_prim = stage.GetPrimAtPath(link_path)
        if not link_prim.IsValid():
            continue
        for prim in Usd.PrimRange(link_prim):
            if prim.IsA(UsdGeom.Cube) and "cavity_collision" in str(prim.GetPath()):
                snapshots.append(
                    (str(prim.GetPath()), cache.GetLocalToWorldTransform(prim), link_path)
                )
    return snapshots


def _restore_cavity_cubes(
    stage: Usd.Stage,
    snapshots: list[tuple[str, Gf.Matrix4d, str]],
    anchor_roots: dict[str, Gf.Vec3d],
    root_world: Gf.Matrix4d,
) -> None:
    root_inv = root_world.GetInverse()
    for cube_path, world_xf, link_path in snapshots:
        anchor = anchor_roots.get(link_path)
        if anchor is None:
            continue
        cube_in_root = root_inv * world_xf
        anchor_mat = Gf.Matrix4d(1.0)
        anchor_mat.SetTranslateOnly(Gf.Vec3d(anchor[0], anchor[1], anchor[2]))
        local_xf = anchor_mat.GetInverse() * cube_in_root
        cube_prim = stage.GetPrimAtPath(cube_path)
        if not cube_prim.IsValid():
            continue
        _strip_xform_ops(cube_prim)
        UsdGeom.Xformable(cube_prim).AddTransformOp(UsdGeom.XformOp.PrecisionDouble).Set(local_xf)


def _freeze_animations_at_time_zero(stage: Usd.Stage) -> None:
    """Collapse animated xforms to t=0 so static merge uses closed drawer pose."""
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Xformable):
            continue
        for attr in prim.GetAttributes():
            if not attr.GetName().startswith("xformOp:"):
                continue
            samples = attr.GetTimeSamples()
            if not samples:
                continue
            value = attr.Get(Usd.TimeCode(0))
            attr.Clear()
            attr.Set(value)


def _xform_cache() -> UsdGeom.XformCache:
    return UsdGeom.XformCache(Usd.TimeCode(0))


def _hide_cavity_collision(stage: Usd.Stage) -> None:
    for prim in stage.Traverse():
        if "cavity_collision" not in str(prim.GetPath()):
            continue
        if prim.IsA(UsdGeom.Imageable):
            UsdGeom.Imageable(prim).MakeInvisible()


def _snapshot_joint_anchors(stage: Usd.Stage, root_world: Gf.Matrix4d) -> list[tuple[str, Gf.Vec3d]]:
    cache = _xform_cache()
    root_inv = root_world.GetInverse()
    anchors: list[tuple[str, Gf.Vec3d]] = []
    for prim in stage.Traverse():
        if not prim.IsA(UsdPhysics.Joint):
            continue
        body0 = UsdPhysics.Joint(prim).GetBody0Rel().GetTargets()
        if not body0:
            continue
        link0 = stage.GetPrimAtPath(body0[0])
        w0 = cache.GetLocalToWorldTransform(link0)
        pos0 = prim.GetAttribute("physics:localPos0").Get()
        anchor_w = w0.Transform(Gf.Vec3d(pos0[0], pos0[1], pos0[2]))
        anchors.append((str(prim.GetPath()), root_inv.Transform(anchor_w)))
    return anchors


def _set_top_drawer_joint_friction(stage: Usd.Stage, friction: float = 5.0) -> None:
    prim = stage.GetPrimAtPath(TOP_JOINT)
    if prim.IsValid():
        prim.GetAttribute("physxJoint:jointFriction").Set(friction)


def _rewrite_joints(stage: Usd.Stage, joint_anchors: list[tuple[str, Gf.Vec3d]], root_world: Gf.Matrix4d) -> None:
    cache = _xform_cache()
    drawer_links = {TOP_DRAWER_LINK}
    for joint_path, anchor_root in joint_anchors:
        if joint_path == BOTTOM_JOINT:
            continue
        prim = stage.GetPrimAtPath(joint_path)
        if not prim.IsValid():
            continue
        body0 = UsdPhysics.Joint(prim).GetBody0Rel().GetTargets()
        body1 = UsdPhysics.Joint(prim).GetBody1Rel().GetTargets()
        body1_path = str(body1[0]) if body1 else ""
        anchor_world = _root_to_world(root_world, anchor_root)
        if body0:
            body0_prim = stage.GetPrimAtPath(body0[0])
            pos0 = _world_to_body_local(body0_prim, cache, anchor_world)
        else:
            pos0 = Gf.Vec3f(anchor_root[0], anchor_root[1], anchor_root[2])
        pos1 = Gf.Vec3f(0.0, 0.0, 0.0) if body1_path in drawer_links else pos0
        prim.GetAttribute("physics:localPos0").Set(pos0)
        prim.GetAttribute("physics:localPos1").Set(pos1)


def _joint_anchor_in_root(stage: Usd.Stage, link_path: str, root_world: Gf.Matrix4d) -> Gf.Vec3d | None:
    cache = _xform_cache()
    root_inv = root_world.GetInverse()
    for prim in stage.Traverse():
        if not prim.IsA(UsdPhysics.Joint):
            continue
        body1 = UsdPhysics.Joint(prim).GetBody1Rel().GetTargets()
        if not body1 or str(body1[0]) != link_path:
            continue
        body0 = UsdPhysics.Joint(prim).GetBody0Rel().GetTargets()
        if not body0:
            continue
        link0 = stage.GetPrimAtPath(body0[0])
        w0 = cache.GetLocalToWorldTransform(link0)
        pos0 = prim.GetAttribute("physics:localPos0").Get()
        return root_inv.Transform(w0.Transform(Gf.Vec3d(pos0[0], pos0[1], pos0[2])))
    return None


def _mesh_bbox_center(stage: Usd.Stage, mesh_path: str) -> tuple[float, float, float]:
    prim = stage.GetPrimAtPath(mesh_path)
    if not prim.IsValid() or not prim.IsA(UsdGeom.Mesh):
        return (0.0, 0.0, 0.0)
    mesh = UsdGeom.Mesh(prim)
    pts = mesh.GetPointsAttr().Get()
    cache = _xform_cache()
    world = cache.GetLocalToWorldTransform(prim)
    ws = [world.Transform(Gf.Vec3d(p[0], p[1], p[2])) for p in pts]
    xs = [w[0] for w in ws]
    ys = [w[1] for w in ws]
    zs = [w[2] for w in ws]
    return ((min(xs) + max(xs)) * 0.5, (min(ys) + max(ys)) * 0.5, (min(zs) + max(zs)) * 0.5)


def _remove_physics_schemas(prim: Usd.Prim) -> None:
    for schema in list(prim.GetAppliedSchemas()):
        if "RigidBody" in schema or schema == "PhysicsMassAPI":
            try:
                if schema == "PhysicsMassAPI":
                    prim.RemoveAPI(UsdPhysics.MassAPI)
                elif "RigidBody" in schema:
                    prim.RemoveAPI(UsdPhysics.RigidBodyAPI)
            except Exception:
                pass


def _transform_mesh_to_body_local(
    mesh_prim: Usd.Prim, cache: UsdGeom.XformCache, body_prim: Usd.Prim
) -> None:
    if not mesh_prim.IsA(UsdGeom.Mesh):
        return
    body_inv = cache.GetLocalToWorldTransform(body_prim).GetInverse()
    mesh_world = cache.GetLocalToWorldTransform(mesh_prim)
    rel = body_inv * mesh_world
    mesh = UsdGeom.Mesh(mesh_prim)
    pts = mesh.GetPointsAttr().Get()
    if not pts:
        return
    baked = []
    for p in pts:
        lp = rel.Transform(Gf.Vec3d(p[0], p[1], p[2]))
        baked.append(Gf.Vec3f(lp[0], lp[1], lp[2]))
    _write_mesh_points(mesh_prim, baked)


def _transform_cube_to_body_local(
    cube_prim: Usd.Prim, cache: UsdGeom.XformCache, body_prim: Usd.Prim
) -> None:
    if not cube_prim.IsA(UsdGeom.Cube):
        return
    body_inv = cache.GetLocalToWorldTransform(body_prim).GetInverse()
    cube_world = cache.GetLocalToWorldTransform(cube_prim)
    local_xf = body_inv * cube_world
    _strip_xform_ops(cube_prim)
    UsdGeom.Xformable(cube_prim).AddTransformOp(UsdGeom.XformOp.PrecisionDouble).Set(
        Gf.Matrix4d(local_xf)
    )


def _rewrite_path(path_str: str, old_prefix: str, new_prefix: str) -> str:
    if path_str.startswith(old_prefix):
        return new_prefix + path_str[len(old_prefix):]
    return path_str


def _fix_stale_prim_paths(stage: Usd.Stage) -> None:
    """Update material/connect paths after drawer_02 → static_drawer_02 relocate."""
    old_prefix = f"{ROOT}/toolbox/toolbox/drawer_02/"
    new_prefix = f"{ROOT_LINK}/static_drawer_02/"
    for prim in stage.Traverse():
        for rel in prim.GetRelationships():
            targets = rel.GetTargets()
            if not targets:
                continue
            updated = []
            changed = False
            for target in targets:
                path_str = _rewrite_path(str(target), old_prefix, new_prefix)
                if path_str != str(target):
                    changed = True
                updated.append(Sdf.Path(path_str))
            if changed:
                rel.SetTargets(updated)
        for attr in prim.GetAttributes():
            if not attr.HasValue():
                continue
            val = attr.Get()
            if isinstance(val, Sdf.Path):
                new_val = _rewrite_path(str(val), old_prefix, new_prefix)
                if new_val != str(val):
                    attr.Set(Sdf.Path(new_val))
            elif isinstance(val, str) and old_prefix in val:
                attr.Set(_rewrite_path(val, old_prefix, new_prefix))


def _merge_static_bottom_drawer(stage: Usd.Stage) -> None:
    """Merge bottom drawer into fixed toolbox body; remove drawer_02_joint DOF."""
    body_prim = stage.GetPrimAtPath(ROOT_LINK)
    bottom_prim = stage.GetPrimAtPath(BOTTOM_DRAWER_LINK)
    if not body_prim.IsValid() or not bottom_prim.IsValid():
        return

    if not stage.GetPrimAtPath(STATIC_BOTTOM).IsValid():
        UsdGeom.Xform.Define(stage, STATIC_BOTTOM)
    static_prim = stage.GetPrimAtPath(STATIC_BOTTOM)
    _set_identity_xform(static_prim)

    cache = _xform_cache()
    edits = Sdf.BatchNamespaceEdit()

    for child in list(bottom_prim.GetChildren()):
        old_path = str(child.GetPath())
        name = child.GetName()
        new_path = f"{STATIC_BOTTOM}/{name}"
        if stage.GetPrimAtPath(new_path).IsValid():
            stage.RemovePrim(new_path)
        edits.Add(old_path, new_path)

    if edits:
        stage.GetRootLayer().Apply(edits)

    static_prim = stage.GetPrimAtPath(STATIC_BOTTOM)
    for prim in Usd.PrimRange(static_prim):
        _remove_physics_schemas(prim)
        if prim.IsA(UsdGeom.Mesh):
            _transform_mesh_to_body_local(prim, cache, body_prim)
        elif prim.IsA(UsdGeom.Cube):
            _transform_cube_to_body_local(prim, cache, body_prim)
        elif prim.IsA(UsdGeom.Xformable) and prim != static_prim:
            _set_identity_xform(prim)

    joint_prim = stage.GetPrimAtPath(BOTTOM_JOINT)
    if joint_prim.IsValid():
        stage.RemovePrim(BOTTOM_JOINT)
    stage.RemovePrim(BOTTOM_DRAWER_LINK)
    _fix_stale_prim_paths(stage)


def _add_drawer_handle_top(stage: Usd.Stage) -> tuple[float, float, float]:
    """Franka-style handle frame prim at knob bbox center (drawer link frame)."""
    knob_mesh = stage.GetPrimAtPath(f"{TOP_DRAWER_LINK}/FurnitureKnob_01/Mesh")
    if not knob_mesh.IsValid():
        return (0.0, 0.0, 0.0)

    cx, cy, cz = _mesh_bbox_center_local(knob_mesh)
    handle_path = HANDLE_TOP
    if stage.GetPrimAtPath(handle_path).IsValid():
        stage.RemovePrim(handle_path)
    handle = UsdGeom.Xform.Define(stage, handle_path)
    xf = UsdGeom.Xformable(handle)
    xf.ClearXformOpOrder()
    xf.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(cx, cy, cz))
    return (cx, cy, cz)


def _flatten_toolbox(stage: Usd.Stage) -> dict[str, tuple[float, float, float]]:
    _freeze_animations_at_time_zero(stage)
    cache = _xform_cache()
    root_prim = stage.GetPrimAtPath(ROOT)
    root_world = cache.GetLocalToWorldTransform(root_prim)
    refs = {
        "toolbox_bbox": _mesh_bbox_center(stage, ROOT_LINK),
        "knob1_world": _mesh_bbox_center(stage, f"{TOP_DRAWER_LINK}/FurnitureKnob_01/Mesh"),
    }
    joint_anchors = _snapshot_joint_anchors(stage, root_world)
    cavity_cubes = _snapshot_cavity_cubes(stage)
    anchor_roots: dict[str, Gf.Vec3d] = {}

    for link_path in LINK_PATHS:
        link_prim = stage.GetPrimAtPath(link_path)
        if not link_prim.IsValid():
            continue
        anchor_root = _joint_anchor_in_root(stage, link_path, root_world)
        if anchor_root is not None and link_path in ACTIVE_DRAWER_LINKS:
            anchor_roots[link_path] = anchor_root
        mesh_prims: list[Usd.Prim] = []
        if link_prim.IsA(UsdGeom.Mesh):
            mesh_prims.append(link_prim)
        for prim in Usd.PrimRange(link_prim):
            if prim.IsA(UsdGeom.Mesh) and prim != link_prim:
                mesh_prims.append(prim)
        for mesh_prim in mesh_prims:
            if link_path in (ROOT_LINK, BOTTOM_DRAWER_LINK) or anchor_root is None:
                baked = _mesh_world_points_in_root(mesh_prim, cache, root_world)
                _write_mesh_points(mesh_prim, baked)
            else:
                _bake_mesh_at_joint_anchor(mesh_prim, cache, root_world, anchor_root)

    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if path == ROOT or not path.startswith(f"{ROOT}/"):
            continue
        if prim.IsA(UsdGeom.Xformable):
            _set_identity_xform(prim)

    _restore_cavity_cubes(stage, cavity_cubes, anchor_roots, root_world)
    _hide_cavity_collision(stage)
    _rewrite_joints(stage, joint_anchors, root_world)
    _merge_static_bottom_drawer(stage)
    _set_top_drawer_joint_friction(stage)
    handle_local = _add_drawer_handle_top(stage)
    refs["drawer_handle_top"] = handle_local
    return refs


def _mesh_z_range_in_root(stage: Usd.Stage, mesh_path: str) -> tuple[float, float]:
    prim = stage.GetPrimAtPath(mesh_path)
    if not prim.IsValid():
        return (0.0, 0.0)
    cache = _xform_cache()
    root_world = cache.GetLocalToWorldTransform(stage.GetPrimAtPath(ROOT))
    pts = UsdGeom.Mesh(prim).GetPointsAttr().Get()
    if not pts:
        return (0.0, 0.0)
    mesh_world = cache.GetLocalToWorldTransform(prim)
    root_inv = root_world.GetInverse()
    zs = [
        root_inv.Transform(mesh_world.Transform(Gf.Vec3d(p[0], p[1], p[2])))[2]
        for p in pts
    ]
    return (min(zs), max(zs))


def _closed_top_drawer_z_in_root(stage: Usd.Stage) -> tuple[float, float]:
    """Top drawer world Z in root when drawer_joint is closed (link at joint anchor)."""
    cache = _xform_cache()
    root_world = cache.GetLocalToWorldTransform(stage.GetPrimAtPath(ROOT))
    joint = stage.GetPrimAtPath(TOP_JOINT)
    body0 = stage.GetPrimAtPath(UsdPhysics.Joint(joint).GetBody0Rel().GetTargets()[0])
    pos0 = joint.GetAttribute("physics:localPos0").Get()
    anchor_world = cache.GetLocalToWorldTransform(body0).Transform(Gf.Vec3d(pos0[0], pos0[1], pos0[2]))
    anchor_root = root_world.GetInverse().Transform(anchor_world)
    mesh = stage.GetPrimAtPath(f"{TOP_DRAWER_LINK}/drawer")
    pts = UsdGeom.Mesh(mesh).GetPointsAttr().Get()
    zs = [anchor_root[2] + p[2] for p in pts]
    return (min(zs), max(zs))


def _validate_desk(stage: Usd.Stage, flat_stage: Usd.Stage) -> None:
    """Sanity-check closed top drawer matches flat source and sits above bottom drawer."""
    flat_top_z = _mesh_z_range_in_root(flat_stage, f"{TOP_DRAWER_LINK}/drawer")
    desk_top_z = _closed_top_drawer_z_in_root(stage)
    bottom_z = _mesh_z_range_in_root(stage, f"{STATIC_BOTTOM}/drawer_02")
    overlap = min(desk_top_z[1], bottom_z[1]) - max(desk_top_z[0], bottom_z[0])
    anchor = stage.GetPrimAtPath(TOP_JOINT).GetAttribute("physics:localPos0").Get()
    print(f"  validate flat top drawer Z in root: [{flat_top_z[0]:.4f}, {flat_top_z[1]:.4f}]")
    print(f"  validate desk top drawer Z (closed): [{desk_top_z[0]:.4f}, {desk_top_z[1]:.4f}]")
    print(f"  validate bottom static mesh Z in root: [{bottom_z[0]:.4f}, {bottom_z[1]:.4f}]")
    print(f"  validate Z overlap: {overlap:.4f} m (expect <= 0)")
    print(f"  drawer_joint localPos0: {anchor}")
    if abs(desk_top_z[0] - flat_top_z[0]) > 0.005 or abs(desk_top_z[1] - flat_top_z[1]) > 0.005:
        raise RuntimeError(
            f"Desk top drawer Z [{desk_top_z}] diverged from flat [{flat_top_z}]"
        )
    if overlap > 0.001:
        raise RuntimeError(
            f"Top/bottom drawer Z overlap {overlap:.4f} m after rebake — check fix_toolbox_xform"
        )
    if desk_top_z[0] < bottom_z[1] - 0.005:
        raise RuntimeError(
            f"Top drawer Z [{desk_top_z[0]:.4f}] should be above bottom max Z [{bottom_z[1]:.4f}]"
        )


def _rebake_usd(dst_path: str, flat_ref_path: str) -> dict[str, tuple[float, float, float]]:
    if not os.path.isfile(flat_ref_path):
        raise FileNotFoundError(flat_ref_path)
    backup = flat_ref_path.replace(".usda", "_pre_rebake.usda")
    if not os.path.isfile(backup):
        shutil.copy2(flat_ref_path, backup)
    if dst_path != flat_ref_path:
        shutil.copy2(flat_ref_path, dst_path)
    stage = Usd.Stage.Open(dst_path)
    if stage is None:
        raise RuntimeError(f"Failed to open {dst_path}")
    flat_stage = Usd.Stage.Open(backup)
    refs = _flatten_toolbox(stage)
    _validate_desk(stage, flat_stage)
    stage.GetRootLayer().Save()
    return refs


def main() -> None:
    if not os.path.isfile(SRC):
        raise FileNotFoundError(SRC)
    refs = _rebake_usd(DST_FLAT, SRC)
    # Rebake can mis-place top drawer at bottom tier; restore from pre-rebake reference.
    stage = Usd.Stage.Open(DST_FLAT)
    pre_path = SRC.replace(".usda", "_pre_rebake.usda")
    ref_stage = Usd.Stage.Open(pre_path) if os.path.isfile(pre_path) else None
    dz = apply_tier_shift_to_stage(stage, ref_stage)
    if dz != 0.0:
        print(f"  tier fix: shifted top drawer geometry by dz={dz:.6f} m")
        validate_tier(stage)
        stage.GetRootLayer().Save()
    print(f"Wrote {DST_FLAT}")
    print("  Active joint: drawer_joint only (bottom drawer static)")
    for name, pos in refs.items():
        print(f"  {name}: ({pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f})")
    shutil.copy2(DST_FLAT, DST_DESK)
    print(f"Copied {DST_FLAT} -> {DST_DESK}")


if __name__ == "__main__":
    main()
