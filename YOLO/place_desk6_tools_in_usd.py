#!/usr/bin/env python3
"""USD 씬을 열어 작업대 위에 공구 6개를 고정 배치하고 저장합니다."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

YOLO_DIR = Path(__file__).resolve().parent
if str(YOLO_DIR) not in sys.path:
    sys.path.insert(0, str(YOLO_DIR))

from generate_realscene_replicator_dataset import (
    DEFAULT_TABLE_SURFACE_PATH,
    DEFAULT_TOOLS_ROOT,
    DEFAULT_UP_AXIS,
    DEFAULT_PLANE_U_AXIS,
    DEFAULT_PLANE_V_AXIS,
    PlacementAxes,
    SCENE_TOOLS,
    capture_tool_base_matrices,
    cleanup_conflicting_replicator_prims,
    compute_staging_bounds_from_table,
    configure_renderer_for_still_images,
    estimate_staging_surface_height,
    set_tool_world_pose,
    set_visibility,
    setup_tools,
    wait_for_stage_ready,
)
from preview_desk6_tools_scene import FIXED_TOOL_YAWS, fixed_grid_positions


DEFAULT_SCENE_USD = Path("/home/user/Desktop/with_camera.usd")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="작업대 위 공구 6개 고정 배치 후 USD 저장")
    parser.add_argument("--scene-usd", type=Path, default=DEFAULT_SCENE_USD)
    parser.add_argument("--output-usd", type=Path, help="저장 경로 (기본: --scene-usd 덮어쓰기)")
    parser.add_argument("--tools-root", type=str, default=DEFAULT_TOOLS_ROOT)
    parser.add_argument("--table-surface-path", type=str, default=DEFAULT_TABLE_SURFACE_PATH)
    parser.add_argument("--grid-cols", type=int, default=3)
    parser.add_argument("--grid-rows", type=int, default=2)
    parser.add_argument(
        "--backup",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="덮어쓰기 전 .bak 백업 생성",
    )
    parser.add_argument("--headless", action="store_true", default=True)
    return parser.parse_args()


def place_tools_fixed_grid(
    stage,
    tool_paths: dict[str, str],
    bounds: dict[str, float],
    surface_height: float,
    tool_base_matrices: dict[str, object],
    axes: PlacementAxes,
    Usd,
    UsdGeom,
    *,
    cols: int,
    rows: int,
) -> None:
    ordered_names = [tool["prim_name"] for tool in SCENE_TOOLS if tool["prim_name"] in tool_paths]
    slots = fixed_grid_positions(bounds, len(ordered_names), cols=cols, rows=rows)

    for prim_name, (u, v) in zip(ordered_names, slots):
        prim_path = tool_paths[prim_name]
        yaw = FIXED_TOOL_YAWS.get(prim_name, 0.0)
        set_tool_world_pose(
            stage,
            prim_path,
            u,
            v,
            surface_height,
            yaw,
            tool_base_matrices[prim_name],
            axes,
            Usd,
            UsdGeom,
        )
        set_visibility(stage, prim_path, True)
        print(f"  ✓ {prim_name}: u={u:.3f}, v={v:.3f}, yaw={yaw:.0f}°")


def main() -> None:
    args = parse_args()
    scene_usd = args.scene_usd.resolve()
    output_usd = (args.output_usd or scene_usd).resolve()

    if not scene_usd.is_file():
        raise FileNotFoundError(f"Scene USD not found: {scene_usd}")

    placement_axes = PlacementAxes(DEFAULT_UP_AXIS, DEFAULT_PLANE_U_AXIS, DEFAULT_PLANE_V_AXIS)

    print("=" * 50)
    print("Place 6 tools on desk → save USD")
    print("=" * 50)
    print(f"Input : {scene_usd}")
    print(f"Output: {output_usd}")
    print()

    from isaacsim import SimulationApp

    simulation_app = SimulationApp(
        {"headless": args.headless, "width": 1280, "height": 720}
    )

    import carb
    import omni.usd
    from pxr import Usd, UsdGeom

    configure_renderer_for_still_images(carb)

    print("[1/3] Scene 로드...")
    omni.usd.get_context().open_stage(str(scene_usd))
    wait_for_stage_ready(simulation_app)
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError("Failed to open scene stage")

    cleanup_conflicting_replicator_prims(stage)
    wait_for_stage_ready(simulation_app, max_updates=60)

    tool_paths = setup_tools(stage, args.tools_root)
    staging_bounds = compute_staging_bounds_from_table(stage, args.table_surface_path, placement_axes)
    surface_height = estimate_staging_surface_height(
        stage, args.table_surface_path, "/World/table", placement_axes
    )
    tool_base_matrices = capture_tool_base_matrices(stage, tool_paths, UsdGeom)

    print("[2/3] 공구 6개 고정 배치...")
    print(f"  staging bounds: {staging_bounds}")
    print(f"  surface height: {surface_height:.4f} m")
    place_tools_fixed_grid(
        stage,
        tool_paths,
        staging_bounds,
        surface_height,
        tool_base_matrices,
        placement_axes,
        Usd,
        UsdGeom,
        cols=args.grid_cols,
        rows=args.grid_rows,
    )

    for _ in range(20):
        simulation_app.update()

    if args.backup and output_usd == scene_usd and scene_usd.exists():
        backup_path = scene_usd.with_suffix(scene_usd.suffix + ".bak")
        shutil.copy2(scene_usd, backup_path)
        print(f"  backup: {backup_path}")

    print("[3/3] USD 저장...")
    stage.GetRootLayer().Export(str(output_usd))
    print(f"✓ saved: {output_usd}")

    simulation_app.close()


if __name__ == "__main__":
    main()
