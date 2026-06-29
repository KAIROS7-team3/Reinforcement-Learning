"""Add drawer-knob grasp collision + fix toolbox USD hierarchy.

Intended layout (articulation container ``toolbox/toolbox/``):

    toolbox/                 (fixed rigid body)
      drawer/                (moving rigid body — knob + grasp collision)
      drawer_joint/
      handle/                (carry handle visual — sibling of drawer, NOT under drawer)

Run once from repo root:
    /home/user/miniconda3/envs/env_isaaclab/bin/python scripts/fix_toolbox_handle_grasp.py
"""

from __future__ import annotations

import os
import shutil

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ASSETS = os.path.join(REPO, "assets")

ROOT = "/toolbox_with_handle"
CONTAINER = f"{ROOT}/toolbox/toolbox"
FIXED_LINK = f"{CONTAINER}/toolbox"
DRAWER_LINK = f"{FIXED_LINK}/drawer"
DRAWER_JOINT = f"{FIXED_LINK}/drawer_joint"
CARRY_HANDLE = f"{FIXED_LINK}/handle"
CARRY_HANDLE_MESH = f"{CARRY_HANDLE}/handle"

# Legacy locations before hierarchy fix.
LEGACY_CONTAINER_DRAWER = f"{CONTAINER}/drawer"
LEGACY_CONTAINER_JOINT = f"{CONTAINER}/drawer_joint"
LEGACY_CONTAINER_HANDLE = f"{CONTAINER}/handle"
LEGACY_DRAWER_HANDLE = f"{LEGACY_CONTAINER_DRAWER}/handle"

KNOB_MESH = f"{DRAWER_LINK}/FurnitureKnob_01/Mesh"
GRASP_COLLISION = f"{DRAWER_LINK}/grasp_collision"
GRASP_PINCH_CUBE = f"{GRASP_COLLISION}/bar"
FRICTION_MAT = f"{DRAWER_LINK}/floor_collision_material"
KNOB_CENTER = (-5.090332e-05, -0.11836937, 0.022230722)
PINCH_CUBE_SIZE = (0.14, 0.032, 0.032)


def _backup(path: str) -> None:
    bak = path + ".bak"
    if not os.path.isfile(bak):
        shutil.copy2(path, bak)
        print(f"backup → {bak}")


def _rewrite_path(value: str, old_prefix: str, new_prefix: str) -> str:
    if value == old_prefix:
        return new_prefix
    if value.startswith(old_prefix + "/"):
        return new_prefix + value[len(old_prefix) :]
    return value


def _rewrite_stage_paths(stage: Usd.Stage, old_prefix: str, new_prefix: str) -> None:
    if old_prefix == new_prefix:
        return
    for prim in stage.Traverse():
        for attr in prim.GetAttributes():
            val = attr.Get()
            if isinstance(val, Sdf.Path):
                new_val = _rewrite_path(str(val), old_prefix, new_prefix)
                if new_val != str(val):
                    attr.Set(Sdf.Path(new_val))
            elif isinstance(val, str) and old_prefix in val:
                attr.Set(_rewrite_path(val, old_prefix, new_prefix))
        for rel in prim.GetRelationships():
            targets = rel.GetTargets()
            if not targets:
                continue
            new_targets = []
            changed = False
            for target in targets:
                old = str(target)
                new = _rewrite_path(old, old_prefix, new_prefix)
                new_targets.append(Sdf.Path(new))
                changed = changed or new != old
            if changed:
                rel.SetTargets(new_targets)


def _reparent_preserve_world(stage: Usd.Stage, src_path: str, dst_path: str) -> None:
    src = Sdf.Path(src_path)
    dst = Sdf.Path(dst_path)
    if stage.GetPrimAtPath(dst):
        if stage.GetPrimAtPath(src):
            stage.RemovePrim(src)
        print(f"already at {dst_path}")
        return
    if not stage.GetPrimAtPath(src):
        return

    src_prim = stage.GetPrimAtPath(src)
    dst_parent = stage.GetPrimAtPath(dst.GetParentPath())
    if not dst_parent:
        raise RuntimeError(f"missing parent for {dst_path}")

    world = UsdGeom.Xformable(src_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    parent_world = UsdGeom.Xformable(dst_parent).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    local = world * parent_world.GetInverse()

    layer = stage.GetRootLayer()
    edits = Sdf.BatchNamespaceEdit()
    edits.Add(str(src), str(dst))
    if not layer.Apply(edits):
        raise RuntimeError(f"namespace move failed: {src_path} → {dst_path}")

    new_prim = stage.GetPrimAtPath(dst)
    if new_prim and new_prim.IsA(UsdGeom.Xformable):
        xf = UsdGeom.Xformable(new_prim)
        xf.ClearXformOpOrder()
        xf.AddTransformOp().Set(local)
    print(f"reparented {src_path} → {dst_path}")


def _align_toolbox_hierarchy(stage: Usd.Stage) -> None:
    """Place drawer / joint / carry handle under the fixed toolbox link."""
    if stage.GetPrimAtPath(LEGACY_CONTAINER_DRAWER):
        _reparent_preserve_world(stage, LEGACY_CONTAINER_DRAWER, DRAWER_LINK)
        _rewrite_stage_paths(stage, LEGACY_CONTAINER_DRAWER, DRAWER_LINK)

    if stage.GetPrimAtPath(LEGACY_CONTAINER_JOINT):
        _reparent_preserve_world(stage, LEGACY_CONTAINER_JOINT, DRAWER_JOINT)

    for src in (LEGACY_DRAWER_HANDLE, LEGACY_CONTAINER_HANDLE):
        if stage.GetPrimAtPath(src):
            _reparent_preserve_world(stage, src, CARRY_HANDLE)
            break

    drawer = stage.GetPrimAtPath(DRAWER_LINK)
    if drawer and drawer.IsA(UsdGeom.Xformable):
        # PhysX allows a rigid body below another rigid body only when the child
        # starts a fresh transform stack. Without this, articulation creation fails.
        UsdGeom.Xformable(drawer).SetResetXformStack(True)

    joint = stage.GetPrimAtPath(DRAWER_JOINT)
    if joint:
        body0 = joint.GetRelationship("physics:body0")
        body1 = joint.GetRelationship("physics:body1")
        if body0:
            body0.SetTargets([Sdf.Path(FIXED_LINK)])
        if body1:
            body1.SetTargets([Sdf.Path(DRAWER_LINK)])


def _bind_friction(stage: Usd.Stage, prim: Usd.Prim) -> None:
    mat = stage.GetPrimAtPath(FRICTION_MAT)
    if mat:
        UsdShade.MaterialBindingAPI(prim).Bind(
            UsdShade.Material(mat),
            bindingStrength=UsdShade.Tokens.weakerThanDescendants,
        )


def _enable_mesh_collision(mesh: Usd.Prim, label: str) -> None:
    UsdPhysics.CollisionAPI.Apply(mesh)
    mesh_collision = UsdPhysics.MeshCollisionAPI.Apply(mesh)
    mesh_collision.CreateApproximationAttr("convexHull")
    mesh.CreateAttribute("physics:collisionEnabled", Sdf.ValueTypeNames.Bool).Set(True)
    mesh.SetMetadata(
        "apiSchemas",
        Sdf.TokenListOp.Create(
            [
                "MaterialBindingAPI",
                "PhysicsCollisionAPI",
                "PhysicsMeshCollisionAPI",
                "PhysxCollisionAPI",
            ]
        ),
    )
    print(f"collision convexHull + PhysxCollisionAPI on {label}")


def _strip_carry_handle_collision(mesh: Usd.Prim) -> None:
    for api in (UsdPhysics.CollisionAPI, UsdPhysics.MeshCollisionAPI):
        if mesh.HasAPI(api):
            mesh.RemoveAPI(api)
    if mesh.HasAttribute("physics:collisionEnabled"):
        mesh.GetAttribute("physics:collisionEnabled").Set(False)
    for schema in list(mesh.GetAppliedSchemas()):
        if schema in {"PhysicsCollisionAPI", "PhysicsMeshCollisionAPI", "PhysxCollisionAPI"}:
            mesh.RemoveAppliedSchema(schema)


def _add_carry_handle_collision(stage: Usd.Stage) -> None:
    mesh = stage.GetPrimAtPath(CARRY_HANDLE_MESH)
    if not mesh:
        print(f"skip carry-handle mesh (not found): {CARRY_HANDLE_MESH}")
        return
    _strip_carry_handle_collision(mesh)
    print(f"carry handle visual only: {CARRY_HANDLE_MESH}")


def _restore_knob_collision(stage: Usd.Stage) -> None:
    mesh = stage.GetPrimAtPath(KNOB_MESH)
    if not mesh:
        print(f"skip knob (not found): {KNOB_MESH}")
        return
    _enable_mesh_collision(mesh, KNOB_MESH)
    _bind_friction(stage, mesh)


def _add_pinch_collision_cube(stage: Usd.Stage) -> None:
    parent = stage.GetPrimAtPath(GRASP_COLLISION)
    if not parent:
        UsdGeom.Xform.Define(stage, GRASP_COLLISION)

    bar_path = Sdf.Path(GRASP_PINCH_CUBE)
    if stage.GetPrimAtPath(bar_path):
        stage.RemovePrim(bar_path)

    cube = UsdGeom.Cube.Define(stage, bar_path)
    cube.CreateSizeAttr(1.0)
    cube.CreateVisibilityAttr("invisible")

    sx, sy, sz = PINCH_CUBE_SIZE
    xform = Gf.Matrix4d()
    xform.SetScale(Gf.Vec3d(sx, sy, sz))
    xform.SetTranslateOnly(Gf.Vec3d(*KNOB_CENTER))
    xf = UsdGeom.Xformable(cube.GetPrim())
    xf.ClearXformOpOrder()
    xf.AddTransformOp().Set(xform)

    prim = cube.GetPrim()
    UsdPhysics.CollisionAPI.Apply(prim)
    prim.CreateAttribute("physics:collisionEnabled", Sdf.ValueTypeNames.Bool).Set(True)
    prim.SetMetadata(
        "apiSchemas",
        Sdf.TokenListOp.Create(["PhysicsCollisionAPI", "PhysxCollisionAPI"]),
    )
    _bind_friction(stage, prim)
    print(
        f"invisible pinch cube at {GRASP_PINCH_CUBE} "
        f"(center={KNOB_CENTER}, size={PINCH_CUBE_SIZE})"
    )


def _patch_usd(path: str) -> None:
    if not os.path.isfile(path):
        print(f"skip (not found): {path}")
        return
    _backup(path)
    stage = Usd.Stage.Open(path)
    _align_toolbox_hierarchy(stage)
    _add_carry_handle_collision(stage)
    _restore_knob_collision(stage)
    _add_pinch_collision_cube(stage)
    stage.GetRootLayer().Save()
    print(f"saved {path}")


def main() -> None:
    for name in ("toolbox_rl_flat.usda", "toolbox_rl_desk.usda"):
        _patch_usd(os.path.join(ASSETS, name))


if __name__ == "__main__":
    main()
