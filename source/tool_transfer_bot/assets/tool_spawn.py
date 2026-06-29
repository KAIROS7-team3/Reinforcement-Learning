"""Spawn visual-only tool USDZ with PhysX rigid body + mesh collision.

``assets/tools/*.usdz`` are normalized for YOLO/replicator (mesh only, no physics).
Isaac Lab ``UsdFileCfg`` only *modifies* existing RigidBodyAPI prims, so we define
physics explicitly after reference spawn.
"""

from __future__ import annotations

from collections.abc import Callable

from isaaclab.sim import schemas
from isaaclab.sim.spawners import materials
from isaaclab.sim.spawners.from_files.from_files import _spawn_from_usd_file
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.sim.utils import bind_physics_material, clone, get_all_matching_child_prims, get_current_stage
from isaaclab.utils import configclass
from pxr import UsdGeom, UsdPhysics


def _apply_tool_mesh_collision(prim_path: str, collision_props) -> None:
    """Convex-hull collision on every mesh under the tool root (not just the first)."""
    stage = get_current_stage()
    mesh_prims = get_all_matching_child_prims(
        prim_path,
        predicate=lambda prim: prim.IsA(UsdGeom.Mesh),
        traverse_instance_prims=True,
    )
    if not mesh_prims:
        return

    for mesh_prim in mesh_prims:
        target_path = str(mesh_prim.GetPath())
        mesh_collision = UsdPhysics.MeshCollisionAPI.Apply(mesh_prim)
        mesh_collision.GetApproximationAttr().Set("convexHull")
        if collision_props is not None:
            schemas.define_collision_properties(target_path, collision_props, stage=stage)


@clone
def spawn_tool_from_usd(
    prim_path: str,
    cfg: UsdFileCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
):
    """Reference tool USDZ then add RigidBodyAPI + mesh collision for ``RigidObject``."""
    prim = _spawn_from_usd_file(prim_path, cfg.usd_path, cfg, translation, orientation, **kwargs)

    if cfg.mass_props is not None:
        schemas.define_mass_properties(prim_path, cfg.mass_props)
    if cfg.rigid_props is not None:
        schemas.define_rigid_body_properties(prim_path, cfg.rigid_props)
    _apply_tool_mesh_collision(prim_path, cfg.collision_props)
    if cfg.physics_material is not None:
        if not cfg.physics_material_path.startswith("/"):
            material_path = f"{prim_path}/{cfg.physics_material_path}"
        else:
            material_path = cfg.physics_material_path
        cfg.physics_material.func(material_path, cfg.physics_material)
        stage = get_current_stage()
        for mesh_prim in get_all_matching_child_prims(
            prim_path,
            predicate=lambda prim: prim.IsA(UsdGeom.Mesh),
            traverse_instance_prims=True,
        ):
            bind_physics_material(str(mesh_prim.GetPath()), material_path, stage=stage)

    return prim


@configclass
class ToolUsdFileCfg(UsdFileCfg):
    """UsdFileCfg that adds physics to visual-only tool meshes."""

    physics_material_path: str = "physicsMaterial"
    """Path to the physics material relative to the tool root (or absolute)."""

    physics_material: materials.RigidBodyMaterialCfg | None = None
    """Friction/restitution override bound to all tool mesh collision prims."""

    func: Callable = spawn_tool_from_usd
