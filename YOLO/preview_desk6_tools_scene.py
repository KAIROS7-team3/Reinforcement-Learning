#!/usr/bin/env python3
"""
랜덤화 없이 작업대 위 공구 6개만 올려둔 씬을 Isaac Sim GUI로 확인합니다.

- 공구함/로봇 숨김
- 공구 6개 고정 그리드 배치 (매번 동일)
- Replicator 캡처 없음 — 뷰어만 열고 대기
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

YOLO_DIR = Path(__file__).resolve().parent
if str(YOLO_DIR) not in sys.path:
    sys.path.insert(0, str(YOLO_DIR))

from generate_realscene_replicator_dataset import (
    DEFAULT_CAMERA_PATH,
    DEFAULT_ROBOT_PATH,
    DEFAULT_SCENE_USD,
    DEFAULT_TABLE_SURFACE_PATH,
    DEFAULT_TOOLBOX_ROOT,
    DEFAULT_TOOLS_ROOT,
    DEFAULT_UP_AXIS,
    DEFAULT_PLANE_U_AXIS,
    DEFAULT_PLANE_V_AXIS,
    PlacementAxes,
    SCENE_TOOLS,
    apply_semantic_labels,
    capture_tool_base_matrices,
    cleanup_conflicting_replicator_prims,
    compute_staging_bounds_from_table,
    configure_renderer_for_still_images,
    create_flat_capture_camera,
    estimate_staging_surface_height,
    hide_prim,
    prepare_scene_for_offline_capture,
    resolve_camera_path,
    set_tool_world_pose,
    set_visibility,
    setup_tools,
    wait_for_stage_ready,
)


# 공구별 고정 yaw (도). 필요하면 여기만 수정.
FIXED_TOOL_YAWS = {
    "Allen_Key_Tool_Assembly": 0.0,
    "Husky_Socket_Wrench": 0.0,
    "Paper_Cutter": 90.0,
    "Screw_Driver": 0.0,
    "Spanner_16mm": 90.0,
    "socket": 0.0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="작업대 + 공구 6개 고정 씬 프리뷰 (GUI)")
    parser.add_argument("--scene-usd", type=Path, default=DEFAULT_SCENE_USD)
    parser.add_argument("--camera-path", type=str, default=DEFAULT_CAMERA_PATH)
    parser.add_argument("--tools-root", type=str, default=DEFAULT_TOOLS_ROOT)
    parser.add_argument("--table-surface-path", type=str, default=DEFAULT_TABLE_SURFACE_PATH)
    parser.add_argument("--flat-camera", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--grid-cols", type=int, default=3, help="고정 배치 그리드 열 수")
    parser.add_argument("--grid-rows", type=int, default=2, help="고정 배치 그리드 행 수")
    return parser.parse_args()


def fixed_grid_positions(
    bounds: dict[str, float],
    count: int,
    *,
    cols: int,
    rows: int,
) -> list[tuple[float, float]]:
    if count > cols * rows:
        raise ValueError(f"grid {cols}x{rows} cannot fit {count} tools")

    u_span = bounds["max_u"] - bounds["min_u"]
    v_span = bounds["max_v"] - bounds["min_v"]
    cell_u = u_span / cols
    cell_v = v_span / rows

    positions: list[tuple[float, float]] = []
    for index in range(count):
        row = index // cols
        col = index % cols
        u = bounds["min_u"] + (col + 0.5) * cell_u
        v = bounds["max_v"] - (row + 0.5) * cell_v
        positions.append((u, v))
    return positions


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


def focus_viewport_on_camera(camera_path: str) -> None:
    try:
        import omni.kit.viewport.utility as vp_utils

        viewport = vp_utils.get_active_viewport()
        if viewport is not None:
            viewport.camera_path = camera_path
            print(f"✓ viewport camera → {camera_path}")
    except Exception as exc:
        print(f"⚠ viewport camera 설정 실패 (수동으로 카메라 선택): {exc}")


def main() -> None:
    args = parse_args()
    args.scene_usd = args.scene_usd.resolve()
    if not args.scene_usd.is_file():
        raise FileNotFoundError(f"Scene USD not found: {args.scene_usd}")

    placement_axes = PlacementAxes(DEFAULT_UP_AXIS, DEFAULT_PLANE_U_AXIS, DEFAULT_PLANE_V_AXIS)

    print("=" * 50)
    print("Desk6 Tools Scene Preview (no randomization)")
    print("=" * 50)
    print(f"Scene: {args.scene_usd}")
    print("GUI를 닫으면 종료됩니다.")
    print()

    from isaacsim import SimulationApp

    simulation_app = SimulationApp({"headless": False, "width": 1600, "height": 900})

    import carb
    import omni.usd
    from pxr import Usd, UsdGeom

    configure_renderer_for_still_images(carb)

    print("[1/4] Scene 로드...")
    omni.usd.get_context().open_stage(str(args.scene_usd))
    wait_for_stage_ready(simulation_app)
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError("Failed to open scene stage")

    cleanup_conflicting_replicator_prims(stage)
    wait_for_stage_ready(simulation_app, max_updates=60)

    camera_prim_path = resolve_camera_path(stage, args.camera_path)
    if args.flat_camera:
        camera_prim_path = create_flat_capture_camera(stage, camera_prim_path)

    tool_paths = setup_tools(stage, args.tools_root)
    staging_bounds = compute_staging_bounds_from_table(stage, args.table_surface_path, placement_axes)
    surface_height = estimate_staging_surface_height(
        stage, args.table_surface_path, "/World/table", placement_axes
    )

    hide_prim(stage, DEFAULT_ROBOT_PATH)
    hide_prim(stage, DEFAULT_TOOLBOX_ROOT)
    apply_semantic_labels(stage, tool_paths)
    prepare_scene_for_offline_capture(stage, tool_paths)
    tool_base_matrices = capture_tool_base_matrices(stage, tool_paths, UsdGeom)

    print("[2/4] 공구 6개 고정 배치...")
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

    print("[3/4] 뷰포트 카메라 설정...")
    for _ in range(30):
        simulation_app.update()
    focus_viewport_on_camera(camera_prim_path)

    print("[4/4] 프리뷰 실행 중...")
    print("  - 마우스로 뷰 회전/이동 가능")
    print("  - 종료: Isaac Sim 창 닫기")
    while simulation_app.is_running():
        simulation_app.update()

    simulation_app.close()


if __name__ == "__main__":
    main()
